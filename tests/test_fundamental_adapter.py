# -*- coding: utf-8 -*-
"""
Tests for fundamental adapter helpers.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    _build_dividend_payload,
    _extract_cash_dividend_per_share,
    _extract_financial_abstract,
    _extract_latest_row,
    _is_vertical_layout,
    _parse_dividend_plan_to_per_share,
    _pick_by_keywords,
    _pick_cash_dividend_ratio_per_10,
)


class TestFundamentalAdapter(unittest.TestCase):
    def test_parse_dividend_plan_to_per_share_supports_cn_patterns(self) -> None:
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10派3元(含税)"), 0.3, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每10股派发2.5元"), 0.25, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每股派0.8元"), 0.8, places=6)
        self.assertIsNone(_parse_dividend_plan_to_per_share("仅送股，不现金分红"))

    def test_extract_latest_row_returns_none_when_code_mismatch(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["600000", "000001"],
                "值": [1, 2],
            }
        )
        row = _extract_latest_row(df, "600519")
        self.assertIsNone(row)

    def test_extract_latest_row_fallback_when_no_code_column(self) -> None:
        df = pd.DataFrame({"值": [1, 2]})
        row = _extract_latest_row(df, "600519")
        self.assertIsNotNone(row)
        self.assertEqual(row["值"], 1)

    def test_dragon_tiger_no_match_with_code_column_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "日期": ["2026-01-01"],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_on_list"])
        self.assertEqual(result["recent_count"], 0)

    def test_dragon_tiger_match_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "日期": [today],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["is_on_list"])
        self.assertGreaterEqual(result["recent_count"], 1)

    def test_fundamental_bundle_includes_financial_report_and_dividend_payload(self) -> None:
        adapter = AkshareFundamentalAdapter()
        now = datetime.now()
        within_ttm = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        future_day = (now + timedelta(days=10)).strftime("%Y-%m-%d")
        old_day = (now - timedelta(days=500)).strftime("%Y-%m-%d")
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": [within_ttm],
                "营业总收入": [1000.0],
                "归母净利润": [300.0],
                "经营活动产生的现金流量净额": [500.0],
                "净资产收益率": [18.2],
                "营业收入同比": [12.0],
                "净利润同比": [9.5],
            }
        )
        forecast_df = pd.DataFrame({"股票代码": ["600519"], "预告": ["预增"]})
        quick_df = pd.DataFrame({"股票代码": ["600519"], "快报": ["快报摘要"]})
        dividend_df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519", "600519", "600519"],
                "除息日": [within_ttm, within_ttm, future_day, old_day],
                "分配方案": ["10派3元(含税)", "10派3元(含税)", "10派5元", "10派1元"],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (forecast_df, "stock_yjyg_em", []),
                (quick_df, "stock_yjkb_em", []),
                (dividend_df, "stock_fhps_detail_em", []),
                (None, None, []),
                (None, None, []),
            ],
        ):
            result = adapter.get_fundamental_bundle("600519")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), within_ttm)
        self.assertEqual(financial_report.get("revenue"), 1000.0)
        self.assertEqual(financial_report.get("net_profit_parent"), 300.0)
        self.assertEqual(financial_report.get("operating_cash_flow"), 500.0)
        self.assertEqual(financial_report.get("roe"), 18.2)

        dividend_payload = result["earnings"].get("dividend", {})
        events = dividend_payload.get("events", [])
        self.assertEqual(len(events), 2)  # duplicate + future day filtered
        self.assertEqual(dividend_payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(dividend_payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_build_dividend_payload_returns_empty_when_code_not_matched(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "除息日": [now],
                "分配方案": ["10派3元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_skips_after_tax_plan(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "除息日": [now],
                "分配方案": ["10派3元(税后)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_ttm_window_boundary(self) -> None:
        now = datetime.now()
        day_365 = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        day_366 = (now - timedelta(days=366)).strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519"],
                "除息日": [day_365, day_366],
                "分配方案": ["10派3元(含税)", "10派5元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_is_vertical_layout_detects_abstract_shape(self) -> None:
        df = pd.DataFrame(
            {
                "选项": ["常用指标"],
                "指标": ["归母净利润"],
                "20260630": [1.0],
                "20250630": [2.0],
            }
        )
        self.assertTrue(_is_vertical_layout(df))
        horizontal = pd.DataFrame({"股票代码": ["002555"], "报告期": ["2026-06-30"], "营业总收入": [1.0]})
        self.assertFalse(_is_vertical_layout(horizontal))

    def test_extract_financial_abstract_vertical_layout(self) -> None:
        """stock_financial_abstract returns one metric per row, periods as columns."""
        df = pd.DataFrame(
            {
                "选项": [
                    "常用指标", "常用指标", "常用指标", "常用指标", "常用指标",
                    "成长能力", "成长能力", "盈利能力",
                ],
                "指标": [
                    "归母净利润", "营业总收入", "净利润", "营业成本", "经营现金流量净额",
                    "营业总收入增长率", "归属母公司净利润增长率", "净资产收益率(ROE)",
                ],
                "20260630": [1766269020.71, 7274643508.72, 1.0, 1.0, 1088989342.51, -14.2, 26.1, 12.35],
                "20260331": [1.0, 1.0, 1.0, 1.0, 1.0, -10.0, 5.0, 6.0],
                "20251231": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 20.0],
            }
        )
        abstract = _extract_financial_abstract(df)
        report = abstract["financial_report"]
        growth = abstract["growth"]

        self.assertEqual(report["report_date"], "2026-06-30")
        self.assertAlmostEqual(report["revenue"], 7274643508.72, places=2)
        self.assertAlmostEqual(report["net_profit_parent"], 1766269020.71, places=2)
        self.assertAlmostEqual(report["operating_cash_flow"], 1088989342.51, places=2)
        self.assertAlmostEqual(report["roe"], 12.35, places=2)
        self.assertAlmostEqual(growth["revenue_yoy"], -14.2, places=4)
        self.assertAlmostEqual(growth["net_profit_yoy"], 26.1, places=4)
        self.assertAlmostEqual(growth["roe"], 12.35, places=2)

    def test_extract_financial_abstract_computes_yoy_fallback(self) -> None:
        """When growth-rate rows are absent, YoY is computed from period values."""
        df = pd.DataFrame(
            {
                "选项": ["常用指标"] * 4,
                "指标": ["归母净利润", "营业总收入", "净利润", "营业成本"],
                "20260630": [120.0, 1000.0, 120.0, 1.0],
                "20250630": [100.0, 800.0, 100.0, 1.0],
            }
        )
        abstract = _extract_financial_abstract(df)
        growth = abstract["growth"]
        self.assertAlmostEqual(growth["revenue_yoy"], 25.0, places=4)
        self.assertAlmostEqual(growth["net_profit_yoy"], 20.0, places=4)

    def test_parse_dividend_plan_supports_song_and_pai(self) -> None:
        """10送X派Y pattern (Eastmoney description) parses the cash part."""
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10送10.00派2.00元(含税,扣税后0.80元)"), 0.2, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10转3派5元"), 0.5, places=6)

    def test_cash_dividend_from_eastmoney_ratio_column(self) -> None:
        row = pd.Series(
            {
                "现金分红-现金分红比例": 2.0,
                "现金分红-现金分红比例描述": "10送10.00派2.00元(含税,扣税后0.80元)",
                "预案公告日": pd.NaT,
            }
        )
        self.assertAlmostEqual(_pick_cash_dividend_ratio_per_10(row), 2.0, places=6)
        self.assertAlmostEqual(_extract_cash_dividend_per_share(row), 0.2, places=6)

    def test_pick_by_keywords_skips_nat(self) -> None:
        row = pd.Series({"预案公告日": pd.NaT, "方案进度": "实施分配"})
        val = _pick_by_keywords(row, ["分配方案", "分红方案", "实施方案", "派息方案", "方案", "预案", "方案说明", "现金分红比例"])
        self.assertEqual(val, "实施分配")

    def test_fundamental_bundle_vertical_abstract_end_to_end(self) -> None:
        adapter = AkshareFundamentalAdapter()
        fin_df = pd.DataFrame(
            {
                "选项": [
                    "常用指标", "常用指标", "常用指标", "常用指标", "常用指标",
                    "成长能力", "成长能力", "盈利能力",
                ],
                "指标": [
                    "归母净利润", "营业总收入", "净利润", "营业成本", "经营现金流量净额",
                    "营业总收入增长率", "归属母公司净利润增长率", "净资产收益率(ROE)",
                ],
                "20260630": [1766269020.71, 7274643508.72, 1.0, 1.0, 1088989342.51, -14.2, 26.1, 12.35],
                "20250630": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 11.0],
            }
        )
        dividend_df = pd.DataFrame(
            {
                "股票代码": ["002555"],
                "除息日": ["2026-06-02"],
                "分配方案": ["10派2.1元(含税)"],
            }
        )
        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (None, None, []),  # forecast: unavailable
                (None, None, []),  # quick report: unavailable
                (dividend_df, "stock_fhps_detail_em", []),
                (None, None, []),
                (None, None, []),
            ],
        ):
            result = adapter.get_fundamental_bundle("002555")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), "2026-06-30")
        self.assertAlmostEqual(financial_report.get("revenue"), 7274643508.72, places=2)
        self.assertAlmostEqual(financial_report.get("net_profit_parent"), 1766269020.71, places=2)
        self.assertEqual(result["growth"].get("revenue_yoy"), -14.2)

        dividend_payload = result["earnings"].get("dividend", {})
        events = dividend_payload.get("events", [])
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["cash_dividend_per_share"], 0.21, places=6)


if __name__ == "__main__":
    unittest.main()
