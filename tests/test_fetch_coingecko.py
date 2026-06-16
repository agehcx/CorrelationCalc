import os
import unittest
from unittest.mock import Mock, patch

import requests

import hedge_metrics


class FetchCoinGeckoTests(unittest.TestCase):
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True), patch("hedge_metrics.requests.get") as get_mock:
            with self.assertRaisesRegex(RuntimeError, "COINGECKO_API_KEY is required"):
                hedge_metrics.fetch_coingecko("BTCUSDT", 1_700_000_000_000, 1_700_360_000_000, 30)
            get_mock.assert_not_called()

    def test_uses_pro_endpoint_and_header(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"prices": [[1_700_000_000_000, 45_000.0]]}

        with patch.dict(os.environ, {"COINGECKO_API_KEY": "test-key"}, clear=True), patch(
            "hedge_metrics.requests.get", return_value=response
        ) as get_mock:
            df = hedge_metrics.fetch_coingecko("BTCUSDT", 1_700_000_000_000, 1_700_360_000_000, 30)

        self.assertEqual(len(df), 1)
        self.assertEqual(float(df.iloc[0]["close"]), 45_000.0)
        get_mock.assert_called_once()
        called_url = get_mock.call_args.kwargs["url"] if "url" in get_mock.call_args.kwargs else get_mock.call_args.args[0]
        self.assertTrue(called_url.startswith("https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"))
        headers = get_mock.call_args.kwargs["headers"]
        self.assertEqual(headers["x-cg-pro-api-key"], "test-key")

    def test_retries_transient_http_errors(self):
        retryable_response = Mock()
        retryable_response.status_code = 503
        retryable_error = requests.exceptions.HTTPError("503 Server Error", response=retryable_response)

        first = Mock()
        first.raise_for_status.side_effect = retryable_error

        second = Mock()
        second.raise_for_status.return_value = None
        second.json.return_value = {"prices": [[1_700_000_000_000, 46_000.0]]}

        with patch.dict(os.environ, {"COINGECKO_API_KEY": "test-key"}, clear=True), patch(
            "hedge_metrics.requests.get", side_effect=[first, second]
        ) as get_mock, patch("hedge_metrics.time.sleep") as sleep_mock:
            df = hedge_metrics.fetch_coingecko("BTCUSDT", 1_700_000_000_000, 1_700_360_000_000, 30)

        self.assertEqual(len(df), 1)
        self.assertEqual(get_mock.call_count, 2)
        sleep_mock.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
