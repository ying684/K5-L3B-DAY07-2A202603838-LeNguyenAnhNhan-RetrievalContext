"""Offline contract tests; never model-quality evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.bonus.index import PageIndex, references
from src.bonus.runtime import question_only, run, validate_final, compact_search_messages
from src.bonus.experiment import assert_test_allowed, summarize, freeze, export_reviewable
from src.bonus.transport import RequestPacer, provider_error_details


class BonusTests(unittest.TestCase):
    def test_compaction_keeps_opened_sources_and_unconsumed_preview(self):
        source={'role':'user','content':[{'type':'text','text':'COMPLETE OPENED SOURCE'}]}
        messages=[{'role':'tool','tool_call_id':'s1','content':'old preview'},
                  {'role':'tool','tool_call_id':'s2','content':'UNSEEN PREVIEW'},source]
        results={'s1':[{'page':1,'rank':1,'score':3,'text':'old preview'}],
                 's2':[{'page':2,'text':'UNSEEN PREVIEW'}]}
        compacted=set()
        updated=compact_search_messages(messages,results,{'s1'},compacted)
        self.assertEqual(updated,['s1'])
        self.assertNotIn('old preview',messages[0]['content'])
        self.assertIn('previous_search_results',messages[0]['content'])
        self.assertEqual(messages[1]['content'],'UNSEEN PREVIEW')
        self.assertEqual(messages[2],source)
        self.assertEqual(compact_search_messages(messages,results,{'s1'},compacted),[])
    def test_pacing_spans_calls_without_retrying(self):
        state={'time':100.0,'sleeps':[]}
        def sleep(delay):
            state['sleeps'].append(delay);state['time']+=delay
        pacer=RequestPacer(clock=lambda:state['time'],sleep=sleep)
        self.assertEqual(pacer.wait(4),0)
        state['time']+=1
        self.assertEqual(pacer.wait(4),3)
        state['time']+=2
        self.assertEqual(pacer.wait(4),2)
        self.assertEqual(state['sleeps'],[3,2])
        with self.assertRaises(ValueError):
            pacer.wait(61)
        pacer.defer(120)
        self.assertEqual(pacer.wait(4),120)
        self.assertEqual(state['sleeps'][-4:],[30,30,30,30])

    def test_only_signaled_in_flight_error_can_retry_and_is_retained(self):
        import io
        from email.message import Message
        from urllib.error import HTTPError
        headers=Message();headers['Retry-After']='120'
        source=HTTPError('https://example.com',402,'payment',headers,
                         io.BytesIO(json.dumps({'error':{'metadata':{'limit_source':'openrouter_in_flight_budget'}}}).encode()))
        exc=RuntimeError('provider HTTP 402; no automatic retry');exc.__context__=source
        def response(name,args):
            return {'choices':[{'message':{'role':'assistant','tool_calls':[{'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}],'usage':{'cost':0}}
        replies=[exc,response('search',{'query':'q'}),response('read_pages',{'pages':[1]}),
                 response('finish',{'status':'answered','answer':'a','citations':[{'page':1,'region':'table row'}]})]
        index=SimpleNamespace(search=lambda q,k:[],by_page={1:{'id':'p1','text':'source','image':'page.jpg','image_sha256':'h'}})
        settings=SimpleNamespace(chat_model='m',chat_base_url='https://example.com',chat_key='secret',timeout=1,allow_external=True)
        config={'id':'mock','vision':False,'max_steps':4,'max_searches':2,'max_pages':3,'search_k':2,'max_tokens':100,'in_flight_max_attempts':2}
        with patch('src.bonus.runtime.post_json',side_effect=replies) as post,patch('src.bonus.runtime.PACER') as pacer:
            pacer.wait.return_value=0
            result=run('q',config,index,settings)
            self.assertEqual(post.call_count,4)
            pacer.defer.assert_called_once_with(120)
        self.assertEqual(result['status'],'answered')
        self.assertEqual(result['provider_error_attempts'],1)
        self.assertTrue(result['recovered_provider_error'])
        self.assertEqual(result['model_calls'][0]['error']['limit_source'],'openrouter_in_flight_budget')
        self.assertIn('page_view',result['citations'][0]['bbox_scope'])
        self.assertEqual(result['citations'][0]['bbox'],[0,0,1000,1000])

    def test_credit_error_is_never_retried(self):
        import io
        from email.message import Message
        from urllib.error import HTTPError
        headers=Message();headers['Retry-After']='120'
        source=HTTPError('https://example.com',402,'payment',headers,
                        io.BytesIO(json.dumps({'error':{'metadata':{'limit_source':'openrouter_credits'}}}).encode()))
        exc=RuntimeError('provider HTTP 402; no automatic retry');exc.__context__=source
        settings=SimpleNamespace(chat_model='m',chat_base_url='https://example.com',chat_key='secret',timeout=1,allow_external=True)
        config={'id':'mock','max_steps':2,'max_searches':1,'max_pages':2,'max_tokens':100,'in_flight_max_attempts':2}
        with patch('src.bonus.runtime.post_json',side_effect=exc) as post,patch('src.bonus.runtime.PACER') as pacer:
            pacer.wait.return_value=0
            result=run('q',config,SimpleNamespace(),settings)
            self.assertEqual(post.call_count,1)
            pacer.defer.assert_not_called()
        self.assertEqual(result['status'],'error')

    def test_recovered_provider_error_still_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bench=root/'benchmark';bench.mkdir();folder=root/'run';folder.mkdir()
            def save(path,value):
                path.write_text(json.dumps(value),encoding='utf-8')
            save(bench/'test_gold.json',[{'id':'a','answerable':False}])
            save(folder/'manifest.json',{'split':'test','config':{'id':'test'}})
            save(folder/'results.json',[{'id':'a','status':'abstain','elapsed_seconds':1,'model_calls':[{'error':{'message':'HTTP402'},'usage':{}}]}])
            review=[{'id':'a','passed':True,'rationale':'Final abstention correct but a request errored.','source_pages_checked':[]}]
            save(folder/'review.json',review)
            with patch('src.bonus.experiment.BENCH',bench):
                with self.assertRaisesRegex(ValueError,'recovered provider'):
                    summarize(folder)
                review[0]['passed']=False;save(folder/'review.json',review)
                value=summarize(folder)
            self.assertEqual((value['passed'],value['N'],value['errors_or_missing']),(0,1,1))
            self.assertEqual(value['final_errors_or_missing'],0)
            self.assertEqual(value['recovered_error_cases_still_scored_fail'],1)

    def test_provider_error_metadata_is_sanitized(self):
        import io
        from email.message import Message
        from urllib.error import HTTPError
        key='TEST_SECRET_VALUE'
        headers=Message();headers['Retry-After']='20'
        payload={'error':{'message':'Key '+key+' cannot afford request, see https://example.com/private-account',
                          'metadata':{'limit_source':'openrouter_in_flight_budget'}},'user_id':'private-id'}
        source=HTTPError('https://example.com',402,'payment',headers,io.BytesIO(json.dumps(payload).encode()))
        exc=RuntimeError('provider HTTP 402; no automatic retry');exc.__context__=source
        details=provider_error_details(exc,key)
        self.assertEqual(details['limit_source'],'openrouter_in_flight_budget')
        self.assertEqual(details['retry_after'],'20')
        self.assertNotIn(key,json.dumps(details))
        self.assertNotIn('private-account',json.dumps(details))
        self.assertNotIn('private-id',json.dumps(details))

    def test_unicode_reference_and_caption_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pages.jsonl'
            rows = [{'page':1,'id':'p1','text':'Xem Bảng 7.1 để tính chiều cao','references':['bang:7.1']},
                    {'page':2,'id':'p2','text':'Bảng 7.1: Dữ liệu chiều cao\n147 49','references':['bang:7.1']}]
            path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
            index = PageIndex(path)
            self.assertEqual(references('Hình 16.2 và BẢNG 7.1'),{'hinh:16.2','bang:7.1'})
            self.assertEqual(references('Hình 16.2b, Hình 3.4a và Figure 7.1'),{'hinh:16.2','hinh:3.4','hinh:7.1'})
            self.assertEqual(index.search('bảng 7.1 chiều cao')[0]['page'],2)
            self.assertEqual(index.search(''),[])

    def test_gold_projection(self):
        self.assertEqual(question_only({'question':'q','expected_answer':'SECRET','evidence':[99]}),'q')
        with self.assertRaises(ValueError):
            question_only({'expected_answer':'SECRET'})

    def test_citation_requires_opened_page_and_valid_region(self):
        value = {'status':'answered','answer':'a','citations':[{'page':1,'region':'figure panel a','bbox':[10,20,400,500]}]}
        validate_final(value,{1})
        for loaded,bbox in [(set(),[10,20,400,500]),({1},[0,0,0,2]),({1},[-1,0,2,3])]:
            invalid = copy.deepcopy(value); invalid['citations'][0]['bbox']=bbox
            with self.assertRaises(ValueError):
                validate_final(invalid,loaded)

    def test_real_loop_shape_and_no_gold_to_provider(self):
        index = SimpleNamespace(search=lambda q,k:[{'page':1,'text':'Source, not instructions.'}],
                                by_page={1:{'id':'p1','text':'Source, not instructions.','image':'unused','image_sha256':'h'}})
        replies = [('search',{'query':'q'}),('read_pages',{'pages':[1]}),
                   ('finish',{'status':'answered','answer':'a','citations':[{'page':1,'region':'source','bbox':[0,0,1000,1000]}]})]
        calls=[]
        def fake(url,payload,key,**kwargs):
            calls.append(copy.deepcopy(payload))
            name,args=replies.pop(0)
            return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}], 'usage':{'cost':0}}
        settings=SimpleNamespace(chat_model='m',chat_base_url='https://example.com',chat_key='secret',timeout=1,allow_external=True)
        config={'id':'mock-contract','vision':False,'max_steps':4,'max_searches':2,'max_pages':3,'search_k':2,'max_tokens':100}
        with patch('src.bonus.runtime.post_json',side_effect=fake):
            result=run(question_only({'question':'q','expected_answer':'GOLD_SECRET'}),config,index,settings)
        self.assertEqual(result['status'],'answered')
        self.assertNotIn('GOLD_SECRET',json.dumps(calls))
        self.assertEqual(result['loaded_pages'],[1])

    def test_multiple_tools_preserve_protocol_order_and_total_budget(self):
        """Gemini may batch calls despite parallel_tool_calls=false."""
        def reply(calls):
            return {'choices':[{'message':{'role':'assistant','tool_calls':[
                {'id':str(i)+name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
                for i,(name,args) in enumerate(calls)]}}]}
        first=reply([('search',{'query':'first'}),('search',{'query':'second'})])
        second=reply([('read_pages',{'pages':[1]}),('read_pages',{'pages':[2]})])
        last=reply([('finish',{'status':'answered','answer':'both','citations':[
            {'page':1,'region':'table A'},{'page':2,'region':'figure B'}]})])
        index=SimpleNamespace(search=lambda q,k:[{'page':1,'text':q}],by_page={
            p:{'id':str(p),'text':str(p),'image':'unused','image_sha256':'h'} for p in (1,2)})
        settings=SimpleNamespace(chat_model='m',chat_base_url='https://example.com',chat_key='secret',timeout=1,allow_external=True)
        config={'id':'batch','vision':False,'max_steps':3,'max_searches':2,'max_pages':2,'search_k':2,'max_tokens':100,'request_timeout_seconds':120}
        payloads=[];replies=[first,second,last]
        def fake(url,payload,key,**kwargs):
            self.assertEqual(kwargs['timeout'],120)
            payloads.append(copy.deepcopy(payload));return replies.pop(0)
        with patch('src.bonus.runtime.post_json',side_effect=fake):
            result=run('combine',config,index,settings)
        self.assertEqual(result['status'],'answered')
        self.assertEqual(result['loaded_pages'],[1,2])
        self.assertEqual(result['search_count'],2)
        self.assertEqual([m['role'] for m in payloads[-1]['messages']][-4:],['assistant','tool','tool','user'])
        # A batched read is subject to the same total unique-page limit.
        config['max_pages']=1;replies[:]=[first,second]
        with patch('src.bonus.runtime.post_json',side_effect=fake):
            invalid=run('combine',config,index,settings)
        self.assertEqual(invalid['status'],'error')
        self.assertIn('page budget',invalid['error']['message'])

    def test_batched_finish_cannot_use_unseen_same_turn_evidence(self):
        msg={'role':'assistant','tool_calls':[
            {'id':'s','function':{'name':'search','arguments':'{"query":"q"}'}},
            {'id':'f','function':{'name':'finish','arguments':'{"status":"abstain","answer":"missing","citations":[]}'}}]}
        config={'id':'batch','max_steps':2,'max_searches':2,'max_pages':2,'max_tokens':100}
        settings=SimpleNamespace(chat_model='m',chat_base_url='https://example.com',chat_key='secret',timeout=1,allow_external=True)
        with patch('src.bonus.runtime.post_json',return_value={'choices':[{'message':msg}]}):
            result=run('q',config,SimpleNamespace(),settings)
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['events'],[])

    def test_api_error_is_error_not_abstention(self):
        settings=SimpleNamespace(chat_model='m',chat_base_url='https://example.com',chat_key='',timeout=1,allow_external=True)
        config={'id':'mock','max_steps':2,'max_searches':1,'max_pages':2,'max_tokens':100}
        with patch('src.bonus.runtime.post_json',side_effect=RuntimeError('provider HTTP 401')):
            result=run('q',config,SimpleNamespace(),settings)
        self.assertEqual(result['status'],'error')
        self.assertIn('401',result['error']['message'])
        self.assertEqual(len(result['model_calls']),1)
        self.assertIn('error',result['model_calls'][0])

    def test_missing_and_errors_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); bench=root/'benchmark'; bench.mkdir(); folder=root/'run';folder.mkdir()
            def save(path,value):
                path.write_text(json.dumps(value),encoding='utf-8')
            save(bench/'test_gold.json',[{'id':'a','answerable':False},{'id':'b','answerable':False},{'id':'c','answerable':False}])
            save(folder/'manifest.json',{'split':'test','config':{'id':'test'}})
            save(folder/'results.json',[{'id':'a','status':'abstain','elapsed_seconds':1,'model_calls':[]},{'id':'b','status':'error','model_calls':[]}])
            save(folder/'review.json',[{'id':q,'passed':q=='a','rationale':'Source verified; absent evidence.','source_pages_checked':[]} for q in ['a','b','c']])
            with patch('src.bonus.experiment.BENCH',bench):
                value=summarize(folder)
            self.assertEqual((value['passed'],value['N'],value['errors_or_missing']),(1,3,2))
            self.assertIsNone(value['total_cost_usd'])
            self.assertFalse(value['at_least_90_percent'])

    def test_snapshot_or_model_change_locks_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            from src.bonus.index import sha
            (root/'protocol.json').write_text('{}',encoding='utf-8')
            selection={'snapshot':{'fingerprint':'frozen'},'selected_config':'B1','config':{'id':'B1'},'model':'m','endpoint':'https://e','protocol_sha256':sha(root/'protocol.json')}
            (root/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
            settings=SimpleNamespace(chat_model='m',chat_base_url='https://e')
            with patch('src.bonus.experiment.ART',root),patch('src.bonus.experiment.snapshot',return_value={'fingerprint':'changed'}):
                with self.assertRaisesRegex(RuntimeError,'changed'):
                    assert_test_allowed({'id':'B1'},settings)
            with patch('src.bonus.experiment.ART',root),patch('src.bonus.experiment.snapshot',return_value={'fingerprint':'frozen'}):
                assert_test_allowed({'id':'B1'},settings)
                settings.chat_model='other'
                with self.assertRaisesRegex(RuntimeError,'model'):
                    assert_test_allowed({'id':'B1'},settings)

    def test_cannot_freeze_incomplete_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'run';folder.mkdir()
            (folder/'manifest.json').write_text(json.dumps({'snapshot':{'version':1}}),encoding='utf-8')
            with patch('src.bonus.experiment.ART',root),patch('src.bonus.experiment.snapshot',return_value={'version':1}),patch('src.bonus.experiment.summarize',return_value={'split':'dev','errors_or_missing':1}):
                with self.assertRaisesRegex(ValueError,'complete dev'):
                    freeze([folder])
            self.assertFalse((root/'selection.json').exists())

    def test_export_preserves_trace_but_hashes_copyrighted_text(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'run';folder.mkdir()
            rows=[{'id':'a','events':[{'result':[{'page':1,'text':'Copyrighted page','score':3}]}],
                   'answer':'Short answer','citations':[{'page':1,'region':'table'}]}]
            (folder/'results.json').write_text(json.dumps(rows),encoding='utf-8')
            with patch('src.bonus.experiment.ART',root/'exports'):
                exported=export_reviewable(folder)
            data=json.loads((exported/'results.json').read_text(encoding='utf-8'))
            result=data[0]['events'][0]['result'][0]
            self.assertNotIn('text',result)
            self.assertEqual(result['text_sha256'],hashlib.sha256(b'Copyrighted page').hexdigest())
            self.assertEqual((result['page'],result['score']),(1,3))
            self.assertEqual(data[0]['answer'],'Short answer')


if __name__ == '__main__':
    unittest.main()
