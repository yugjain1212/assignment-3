"""Tests for augmented prompts and LLM response generation."""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.llm import LLMError, build_augmented_prompt, generate_answer
from src.search_engine import SearchResult


class LLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "What protects cloud infrastructure?"
        self.chunks: list[SearchResult] = [
            {
                "page_number": 7,
                "score": 0.8,
                "text": "Cloud security protects cloud infrastructure.",
            }
        ]

    def test_augmented_prompt_contains_query_and_source_context(self) -> None:
        prompt = build_augmented_prompt(self.query, self.chunks)

        self.assertIn("using only the context", prompt)
        self.assertIn("[Page 7]", prompt)
        self.assertIn(self.chunks[0]["text"], prompt)
        self.assertIn(self.query, prompt)

    def test_generate_answer_sends_prompt_to_selected_model(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" Cloud security. "))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=response))
            )
        )
        openai_constructor = Mock(return_value=client)
        openai_module = ModuleType("openai")
        openai_module.__dict__.update(
            OpenAI=openai_constructor, OpenAIError=Exception
        )

        with (
            patch.dict(sys.modules, {"openai": openai_module}),
            patch.dict(
                os.environ,
                {
                    "NVIDIA_API_KEY": "test-key",
                    "BASE_URL": "https://example.test/v1",
                },
            ),
        ):
            prompt, answer = generate_answer(self.query, self.chunks, "test-model")

        openai_constructor.assert_called_once_with(
            api_key="test-key", base_url="https://example.test/v1"
        )
        client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": prompt}],
        )
        self.assertEqual(answer, "Cloud security.")

    def test_missing_api_key_raises_clear_error(self) -> None:
        with patch.dict(
            os.environ,
            {"NVIDIA_API_KEY": "", "OPENAI_API_KEY": "", "API_KEY": ""},
        ):
            with self.assertRaisesRegex(LLMError, "API key is missing"):
                generate_answer(self.query, self.chunks)

    def test_empty_model_response_raises_clear_error(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content="  ")
                                )
                            ]
                        )
                    )
                )
            )
        )
        openai_module = ModuleType("openai")
        openai_module.__dict__.update(
            OpenAI=Mock(return_value=client), OpenAIError=Exception
        )

        with (
            patch.dict(sys.modules, {"openai": openai_module}),
            patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
        ):
            with self.assertRaisesRegex(LLMError, "empty response"):
                generate_answer(self.query, self.chunks)


if __name__ == "__main__":
    unittest.main()
