import copy
import os
import unittest
from unittest.mock import patch

from src.bonus.providers import settings_for,adapt_payload,assistant_input
from src.bonus.runtime import SEARCH,READ,FINISH


class ProviderTests(unittest.TestCase):
    def test_provider_key_selection_has_no_silent_fallback(self):
        with patch('src.config.load_environment'),patch.dict(os.environ,{'GEMINI_API_KEY':'google-secret','OPENAI_API_KEY':'openai-secret','OPENROUTER_API_KEY':'router-secret'},clear=True):
            settings=settings_for({'provider':'gemini','model':'gemini-3.1-flash-lite'})
            self.assertEqual(settings.chat_key,'google-secret')
            self.assertEqual(settings.chat_base_url,'https://generativelanguage.googleapis.com/v1beta/openai')
            self.assertEqual(settings.chat_model,'gemini-3.1-flash-lite')
            with self.assertRaisesRegex(ValueError,'approved'):
                settings_for({'provider':'gemini','model':'gemini-3.1-flash-lite','base_url':'https://wrong.example'})
        with patch('src.config.load_environment'),patch.dict(os.environ,{'OPENROUTER_API_KEY':'router-secret'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'no silent'):
                settings_for({'provider':'openai','model':'gpt-4.1-mini'})

    def test_google_payload_strips_routing_and_preserves_local_schema(self):
        source={'provider':{'require_parameters':True},'tools':[SEARCH,READ,FINISH]}
        original=copy.deepcopy(source)
        result=adapt_payload(source,{'provider':'gemini','reasoning_effort':'minimal'})
        self.assertNotIn('provider',result)
        self.assertFalse(result['parallel_tool_calls'])
        self.assertNotIn('strict',result['tools'][0]['function'])
        self.assertNotIn('minLength',result['tools'][0]['function']['parameters']['properties']['query'])
        self.assertEqual(result['reasoning_effort'],'minimal')
        self.assertEqual(source,original)
        direct=adapt_payload(source,{'provider':'openai'})
        self.assertNotIn('provider',direct)
        self.assertTrue(direct['tools'][0]['function']['strict'])

    def test_thought_signature_survives_tool_roundtrip(self):
        message={'role':'assistant','tool_calls':[{'id':'c','type':'function',
            'function':{'name':'search','arguments':'{}'},'extra_content':{'google':{'thought_signature':'opaque-signature'}}}],
            'annotations':[],'provider_extra':'ignored'}
        result=assistant_input(message,{'provider':'gemini'})
        self.assertEqual(result['tool_calls'][0]['extra_content']['google']['thought_signature'],'opaque-signature')
        self.assertNotIn('annotations',result)
        self.assertNotIn('provider_extra',result)


if __name__=='__main__':
    unittest.main()
