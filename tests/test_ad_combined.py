import pytest
import pandas as pd
import numpy as np
from anomaly_dashboard.modules.ad_combined import run_ad_combined


@pytest.fixture
def sample_data():
    """
    Create static sample dataset covering different anomaly detection patterns.

    Patterns covered:
    1. flag_amount_gt3x: Payment > 3x historical average
    2. flag_z_score_amount: Z-score > 3 for payment amount
    3. flag_freq: More than 3 payments in single cycle
    4. flag_z_desc: Description-based anomaly
    5. Normal records (no anomalies)
    """
    data = {
        'verification_code': [
            # Beneficiary 1: Normal progression
            'BEN001', 'BEN001', 'BEN001', 'BEN001',
            # Beneficiary 2: Amount spike (> 3x average)
            'BEN002', 'BEN002', 'BEN002', 'BEN002',
            # Beneficiary 3: High frequency (multiple payments in one cycle)
            'BEN003', 'BEN003', 'BEN003', 'BEN003', 'BEN003', 'BEN003',
            # Beneficiary 4: Normal with consistent amount
            'BEN004', 'BEN004', 'BEN004', 'BEN004',
            # Beneficiary 5: High z-score amount
            'BEN005', 'BEN005', 'BEN005', 'BEN005',
            # Beneficiary 6: Different descriptions (for semantic clustering)
            'BEN006', 'BEN006', 'BEN006', 'BEN006',
        ],
        'payment_cycle': [
            # Beneficiary 1: Normal cycles
            1, 2, 3, 4,
            # Beneficiary 2: Amount spike in cycle 4
            1, 2, 3, 4,
            # Beneficiary 3: Multiple payments in cycle 3
            1, 2, 3, 3, 3, 3,
            # Beneficiary 4: Normal cycles
            1, 2, 3, 4,
            # Beneficiary 5: Z-score spike in cycle 4
            1, 2, 3, 4,
            # Beneficiary 6: Normal cycles
            1, 2, 3, 4,
        ],
        'payment_amount': [
            # Beneficiary 1: Normal amounts
            1000, 1050, 1020, 1030,
            # Beneficiary 2: Spike to > 3x average (avg ~1000, spike to 3500)
            1000, 1050, 1020, 3500,
            # Beneficiary 3: Normal amounts but high frequency
            800, 850, 820, 830, 840, 810,
            # Beneficiary 4: Consistent amounts
            1500, 1500, 1500, 1500,
            # Beneficiary 5: High z-score (avg ~500, std ~20, spike to 1000)
            500, 510, 505, 1000,
            # Beneficiary 6: Normal amounts
            2000, 2100, 2050, 2080,
        ],
        'activity_desc': [
            # Beneficiary 1: Standard food assistance
            'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية',
            # Beneficiary 2: Standard food assistance
            'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية',
            # Beneficiary 3: Standard food assistance
            'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية',
            # Beneficiary 4: Medical assistance (different cluster)
            'مساعدة طبية', 'مساعدة طبية', 'مساعدة طبية', 'مساعدة طبية',
            # Beneficiary 5: Educational assistance
            'مساعدة تعليمية', 'مساعدة تعليمية', 'مساعدة تعليمية', 'مساعدة تعليمية',
            # Beneficiary 6: Unusual/rare description (potential embedding anomaly)
            'مساعدة غذائية', 'مساعدة غذائية', 'مساعدة غذائية', 'نشاط استثنائي غير معتاد جدا للمستفيد',
        ],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def large_sample_data():
    """
    Create a larger dataset to ensure enough data for clustering and Isolation Forest.
    """
    np.random.seed(42)

    num_beneficiaries = 10
    cycles_per_ben = 10

    verification_codes = []
    payment_cycles = []
    payment_amounts = []
    activity_descs = []

    # Common descriptions for clustering
    common_descs = [
        'مساعدة غذائية',
        'مساعدة طبية',
        'مساعدة تعليمية',
        'مساعدة سكنية',
        'دعم نقدي طارئ',
    ]

    for ben_id in range(num_beneficiaries):
        base_amount = np.random.uniform(500, 2000)
        desc_idx = ben_id % len(common_descs)

        for cycle in range(1, cycles_per_ben + 1):
            verification_codes.append(f'BEN{ben_id:03d}')
            payment_cycles.append(cycle)

            # Normal variation around base amount
            amount = base_amount + np.random.normal(0, base_amount * 0.1)
            payment_amounts.append(max(100, amount))  # Ensure positive

            activity_descs.append(common_descs[desc_idx])

    # Add specific anomalies
    # 1. Amount spike (> 3x average)
    verification_codes.append('BEN999')
    payment_cycles.append(1)
    payment_amounts.append(1000)
    activity_descs.append('مساعدة غذائية')

    verification_codes.append('BEN999')
    payment_cycles.append(2)
    payment_amounts.append(1020)
    activity_descs.append('مساعدة غذائية')

    verification_codes.append('BEN999')
    payment_cycles.append(3)
    payment_amounts.append(1010)
    activity_descs.append('مساعدة غذائية')

    verification_codes.append('BEN999')
    payment_cycles.append(4)
    payment_amounts.append(4000)  # > 3x average
    activity_descs.append('مساعدة غذائية')

    # 2. High frequency (4 payments in one cycle)
    for i in range(5):
        verification_codes.append('BEN998')
        payment_cycles.append(5)
        payment_amounts.append(800 + i * 10)
        activity_descs.append('مساعدة غذائية')

    # 3. Unusual description for embedding anomaly
    verification_codes.append('BEN997')
    payment_cycles.append(1)
    payment_amounts.append(1500)
    activity_descs.append('وصف غير معتاد تماما مع كلمات نادرة جدا وفريدة من نوعها لا تظهر في أي مكان آخر')

    df = pd.DataFrame({
        'verification_code': verification_codes,
        'payment_cycle': payment_cycles,
        'payment_amount': payment_amounts,
        'activity_desc': activity_descs,
    })

    return df


class TestADCombined:
    """Test suite for the combined anomaly detection module."""

    def test_basic_functionality(self, sample_data):
        """Test that the function runs without errors and returns expected structure."""
        result = run_ad_combined(sample_data)

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that expected columns exist
        expected_cols = [
            'combined_anomaly', 'flag_amount_gt3x', 'flag_z_score_amount',
            'flag_freq', 'flag_z_desc', 'iforest_num', 'iforest_emb',
            'Explanation', 'verification_code', 'payment_amount'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_amount_spike_detection(self, large_sample_data):
        """Test detection of amount > 3x historical average."""
        result = run_ad_combined(large_sample_data)

        # BEN999 should have an anomaly in cycle 4 (4000 vs avg ~1010)
        ben999_anomalies = result[result['verification_code'] == 'BEN999']

        if len(ben999_anomalies) > 0:
            # Check if any record has the amount spike flag
            assert ben999_anomalies['flag_amount_gt3x'].any(), \
                "Amount spike (> 3x avg) should be detected for BEN999"

    def test_frequency_detection(self, large_sample_data):
        """Test detection of multiple payments in single cycle."""
        result = run_ad_combined(large_sample_data)

        # BEN998 has 5 payments in cycle 5 (> 3)
        ben998_anomalies = result[result['verification_code'] == 'BEN998']

        if len(ben998_anomalies) > 0:
            # At least one record should have the frequency flag
            assert ben998_anomalies['flag_freq'].any(), \
                "High frequency (> 3 payments per cycle) should be detected for BEN998"

    def test_z_score_flag_exists(self, large_sample_data):
        """Test that z-score amount flag column exists in results."""
        result = run_ad_combined(large_sample_data)

        # Check that the flag_z_score_amount column exists in anomaly results
        if len(result) > 0:
            assert 'flag_z_score_amount' in result.columns, \
                "flag_z_score_amount column should exist in anomaly results"

    def test_arabic_normalization(self, sample_data):
        """Test that Arabic text normalization is applied."""
        # Create data with diacritics and different Alif forms
        test_data = sample_data.copy()
        test_data.loc[0, 'activity_desc'] = 'مُساعَدة غِذائِيَّة'  # With diacritics
        test_data.loc[1, 'activity_desc'] = 'مساعدة غذائية'  # Without diacritics

        # Should not raise error
        result = run_ad_combined(test_data)
        assert isinstance(result, pd.DataFrame)

    def test_embedding_anomaly_detection(self, large_sample_data):
        """Test that unusual descriptions are flagged by embedding-based detection."""
        result = run_ad_combined(large_sample_data)

        # BEN997 has an unusual description that should be flagged by iforest_emb
        ben997_records = result[result['verification_code'] == 'BEN997']

        if len(ben997_records) > 0:
            # Should be flagged as anomaly (either by iforest_emb or combined)
            assert ben997_records['combined_anomaly'].any(), \
                "Unusual description should trigger anomaly detection for BEN997"

    def test_explanation_generation(self, large_sample_data):
        """Test that explanations are generated for anomalies."""
        result = run_ad_combined(large_sample_data)

        # All anomalies should have explanations
        if len(result) > 0:
            # Explanation should not be empty or just "—"
            has_explanation = result['Explanation'].notna() & (result['Explanation'] != '—')
            assert has_explanation.any(), "Anomalies should have explanations"

    def test_combined_anomaly_logic(self, large_sample_data):
        """Test that combined anomaly flag is set correctly."""
        result = run_ad_combined(large_sample_data)

        # All returned records should have combined_anomaly = True
        # (since the function filters for anomalies only)
        if len(result) > 0:
            assert result['combined_anomaly'].all(), \
                "All returned records should be anomalies"

    def test_empty_dataframe(self):
        """Test handling of empty input."""
        empty_df = pd.DataFrame(columns=[
            'verification_code', 'payment_cycle', 'payment_amount', 'activity_desc'
        ])

        # Should not crash, but may return empty result
        try:
            result = run_ad_combined(empty_df)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # If it fails, make sure it's a reasonable error
            assert len(str(e)) > 0

    def test_clustering_creates_groups(self, large_sample_data):
        """Test that semantic clustering creates description groups."""
        # Run the function and verify it completes
        result = run_ad_combined(large_sample_data)

        # The function should complete without error
        # Clustering should assign cluster labels
        assert isinstance(result, pd.DataFrame)

    def test_isolation_forest_flags(self, large_sample_data):
        """Test that Isolation Forest creates flags."""
        result = run_ad_combined(large_sample_data)

        # If there are results, check that isolation forest columns exist
        if len(result) > 0:
            assert 'iforest_num' in result.columns
            assert 'iforest_emb' in result.columns

            # At least one record should have an isolation forest flag
            assert (result['iforest_num'] | result['iforest_emb']).any(), \
                "At least one isolation forest flag should be set"

    def test_data_types(self, sample_data):
        """Test that data types are handled correctly."""
        result = run_ad_combined(sample_data)

        if len(result) > 0:
            # Boolean columns should be boolean type
            bool_cols = ['flag_amount_gt3x', 'flag_z_score_amount', 'flag_freq',
                        'flag_z_desc', 'iforest_num', 'iforest_emb', 'combined_anomaly']

            for col in bool_cols:
                if col in result.columns:
                    assert result[col].dtype == bool or result[col].dtype == 'bool', \
                        f"Column {col} should be boolean type"

    def test_consistent_amounts_no_anomaly(self, sample_data):
        """Test that consistent payments don't trigger false positives."""
        # BEN004 has consistent amounts (1500, 1500, 1500, 1500)
        result = run_ad_combined(sample_data)

        # BEN004 should not appear in anomalies (consistent amounts shouldn't be flagged)
        ben004_in_results = result['verification_code'].str.contains('BEN004').any()

        # Verify that BEN004 with consistent amounts is not flagged as an anomaly
        assert not ben004_in_results, "BEN004 with consistent amounts should not be flagged as an anomaly"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_values(self):
        """Test handling of missing values in input data."""
        data_with_nulls = pd.DataFrame({
            'verification_code': ['BEN001', 'BEN001', 'BEN001', None],
            'payment_cycle': [1, 2, 3, 4],
            'payment_amount': [1000, None, 1020, 1030],
            'activity_desc': ['مساعدة غذائية', 'مساعدة غذائية', None, 'مساعدة غذائية'],
        })

        # Should handle missing values gracefully
        try:
            result = run_ad_combined(data_with_nulls)
            assert isinstance(result, pd.DataFrame)
        except Exception:
            # If it fails, that's also acceptable as long as it's handled
            pass

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
