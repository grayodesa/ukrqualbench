"""Tests for language quality detectors.

Tests cover:
- Base detector functionality
- Russism detection
- Anglicism detection
- Positive marker detection
- Fertility rate calculation
"""


import pytest

from ukrqualbench.detectors import (
    AnglicismDetector,
    DetectionMatch,
    DetectionResult,
    DetectionSeverity,
    FertilityCalculator,
    PositiveMarkerDetector,
    RussismDetector,
    calculate_fertility,
    create_anglicism_detector,
    create_positive_marker_detector,
    create_russism_detector,
    evaluate_fertility_quality,
)

# ============================================================================
# Base Detector Tests
# ============================================================================


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty detection result."""
        result = DetectionResult(text="test", matches=[], total_tokens=100)
        assert result.count == 0
        assert result.rate_per_1k == 0.0
        assert result.weighted_rate_per_1k == 0.0

    def test_rate_calculation(self) -> None:
        """Test rate per 1K tokens calculation."""
        match = DetectionMatch(
            start=0,
            end=5,
            matched_text="test",
            pattern_id="test_001",
            category="test",
            weight=1.0,
        )
        result = DetectionResult(text="test", matches=[match], total_tokens=500)
        assert result.rate_per_1k == 2.0  # 1 match / 500 tokens * 1000

    def test_weighted_rate_calculation(self) -> None:
        """Test weighted rate calculation."""
        match1 = DetectionMatch(
            start=0,
            end=5,
            matched_text="test1",
            pattern_id="test_001",
            category="test",
            weight=2.0,
        )
        match2 = DetectionMatch(
            start=10,
            end=15,
            matched_text="test2",
            pattern_id="test_002",
            category="test",
            weight=3.0,
        )
        result = DetectionResult(text="test", matches=[match1, match2], total_tokens=1000)
        # Total weight = 5, rate = 5/1000 * 1000 = 5.0
        assert result.weighted_rate_per_1k == 5.0

    def test_group_by_category(self) -> None:
        """Test grouping matches by category."""
        match1 = DetectionMatch(
            start=0, end=5, matched_text="t1", pattern_id="p1", category="cat1"
        )
        match2 = DetectionMatch(
            start=10, end=15, matched_text="t2", pattern_id="p2", category="cat2"
        )
        match3 = DetectionMatch(
            start=20, end=25, matched_text="t3", pattern_id="p3", category="cat1"
        )
        result = DetectionResult(text="test", matches=[match1, match2, match3])

        by_cat = result.by_category()
        assert len(by_cat["cat1"]) == 2
        assert len(by_cat["cat2"]) == 1

    def test_group_by_severity(self) -> None:
        """Test grouping matches by severity."""
        match1 = DetectionMatch(
            start=0, end=5, matched_text="t1", pattern_id="p1",
            category="cat", severity=DetectionSeverity.CRITICAL
        )
        match2 = DetectionMatch(
            start=10, end=15, matched_text="t2", pattern_id="p2",
            category="cat", severity=DetectionSeverity.LOW
        )
        result = DetectionResult(text="test", matches=[match1, match2])

        by_sev = result.by_severity()
        assert len(by_sev[DetectionSeverity.CRITICAL]) == 1
        assert len(by_sev[DetectionSeverity.LOW]) == 1


class TestDetectionMatch:
    """Tests for DetectionMatch dataclass."""

    def test_span_property(self) -> None:
        """Test span tuple property."""
        match = DetectionMatch(
            start=10,
            end=20,
            matched_text="test",
            pattern_id="test_001",
            category="test",
        )
        assert match.span == (10, 20)


# ============================================================================
# Russism Detector Tests
# ============================================================================


class TestRussismDetector:
    """Tests for RussismDetector."""

    @pytest.fixture
    def detector(self) -> RussismDetector:
        """Create initialized detector."""
        return create_russism_detector()

    def test_initialization(self, detector: RussismDetector) -> None:
        """Test detector initializes correctly."""
        assert detector._initialized
        assert detector.pattern_count > 0

    def test_detect_pryjniaty_uchast(self, detector: RussismDetector) -> None:
        """Test detection of 'прийняти участь' russism."""
        text = "Ми хочемо прийняти участь у заході."
        result = detector.detect(text)
        assert result.count >= 1
        assert any("участь" in m.matched_text.lower() for m in result.matches)

    def test_detect_miropryjemstvo(self, detector: RussismDetector) -> None:
        """Test detection of 'міроприємство' russism."""
        text = "Це важливе міроприємство для нас."
        result = detector.detect(text)
        assert result.count >= 1
        assert any("міроприємств" in m.matched_text.lower() for m in result.matches)

    def test_detect_na_protyazi(self, detector: RussismDetector) -> None:
        """Test detection of 'на протязі' russism."""
        text = "На протязі всього дня йшов дощ."
        result = detector.detect(text)
        assert result.count >= 1

    def test_detect_yavlyaetsya(self, detector: RussismDetector) -> None:
        """Test detection of 'являється' russism."""
        text = "Це являється основною проблемою."
        result = detector.detect(text)
        assert result.count >= 1

    def test_detect_sliduyuchyj(self, detector: RussismDetector) -> None:
        """Test detection of 'слідуючий' russism."""
        text = "Слідуючий крок дуже важливий."
        result = detector.detect(text)
        assert result.count >= 1

    def test_clean_text_no_matches(self, detector: RussismDetector) -> None:
        """Test that clean text has no matches."""
        text = "Взяти участь у заході дуже важливо."
        result = detector.detect(text)
        # Should not match 'взяти участь'
        assert not any("взяти участь" in m.matched_text.lower() for m in result.matches)

    def test_severity_breakdown(self, detector: RussismDetector) -> None:
        """Test severity breakdown calculation."""
        text = "Прийняти участь у міроприємстві є важливим."
        result = detector.detect(text)
        breakdown = detector.get_severity_breakdown(result)
        assert "critical" in breakdown
        assert isinstance(breakdown["critical"], int)

    def test_quality_score_calculation(self, detector: RussismDetector) -> None:
        """Test quality score calculation."""
        clean_text = "Це чистий український текст без помилок."
        result = detector.detect(clean_text)
        score = detector.calculate_quality_score(result)
        assert 0 <= score <= 100

    def test_get_corrections(self, detector: RussismDetector) -> None:
        """Test getting corrections for russisms."""
        text = "Треба прийняти участь у заході."
        result = detector.detect(text)
        corrections = detector.get_corrections(result)
        if corrections:
            assert "correction" in corrections[0]
            assert "original" in corrections[0]

    def test_detect_with_context(self, detector: RussismDetector) -> None:
        """Test detection with surrounding context."""
        text = "Важливо прийняти участь у цьому заході."
        result = detector.detect_with_context(text, context_chars=20)
        assert "contexts" in result.metadata
        if result.matches:
            assert len(result.metadata["contexts"]) == len(result.matches)

    def test_statistics(self, detector: RussismDetector) -> None:
        """Test detector statistics."""
        stats = detector.get_statistics()
        assert "total_patterns" in stats
        assert "by_category" in stats
        assert "by_severity" in stats
        assert stats["total_patterns"] > 0


# ============================================================================
# Anglicism Detector Tests
# ============================================================================


class TestAnglicismDetector:
    """Tests for AnglicismDetector."""

    @pytest.fixture
    def detector(self) -> AnglicismDetector:
        """Create initialized detector."""
        return create_anglicism_detector()

    def test_initialization(self, detector: AnglicismDetector) -> None:
        """Test detector initializes correctly."""
        assert detector._initialized
        assert detector.pattern_count > 0

    def test_detect_fidbek(self, detector: AnglicismDetector) -> None:
        """Test detection of 'фідбек' anglicism."""
        text = "Чекаю на ваш фідбек щодо проєкту."
        result = detector.detect(text)
        assert result.count >= 1

    def test_detect_dedlayn(self, detector: AnglicismDetector) -> None:
        """Test detection of 'дедлайн' anglicism."""
        text = "Дедлайн вже завтра."
        result = detector.detect(text)
        assert result.count >= 1

    def test_detect_implementuvaty(self, detector: AnglicismDetector) -> None:
        """Test detection of 'імплементувати' anglicism."""
        text = "Потрібно імплементувати цю функцію."
        result = detector.detect(text)
        assert result.count >= 1

    def test_context_aware_detection_technical(self, detector: AnglicismDetector) -> None:
        """Test context-aware detection in technical context."""
        text = "Треба мержити цей коміт."
        result_general = detector.detect_with_context_awareness(text, is_technical=False)
        result_technical = detector.detect_with_context_awareness(text, is_technical=True)

        # In technical context, weights should be lower
        if result_general.matches and result_technical.matches:
            general_weight = sum(m.weight for m in result_general.matches)
            technical_weight = sum(m.weight for m in result_technical.matches)
            assert technical_weight <= general_weight

    def test_category_breakdown(self, detector: AnglicismDetector) -> None:
        """Test category breakdown."""
        text = "Маємо мітинг о 10:00 щодо нової фічі."
        result = detector.detect(text)
        breakdown = detector.get_category_breakdown(result)
        assert isinstance(breakdown, dict)

    def test_quality_score(self, detector: AnglicismDetector) -> None:
        """Test quality score calculation."""
        text = "Це простий текст українською мовою."
        result = detector.detect(text)
        score = detector.calculate_quality_score(result)
        assert 0 <= score <= 100

    def test_get_corrections(self, detector: AnglicismDetector) -> None:
        """Test getting corrections with context notes."""
        text = "Чекаю фідбек після мітингу."
        result = detector.detect(text)
        corrections = detector.get_corrections(result)
        if corrections:
            assert "context_note" in corrections[0]


# ============================================================================
# Positive Marker Detector Tests
# ============================================================================


class TestPositiveMarkerDetector:
    """Tests for PositiveMarkerDetector."""

    @pytest.fixture
    def detector(self) -> PositiveMarkerDetector:
        """Create initialized detector."""
        return create_positive_marker_detector()

    def test_initialization(self, detector: PositiveMarkerDetector) -> None:
        """Test detector initializes correctly."""
        assert detector._initialized
        assert detector.pattern_count > 0

    def test_detect_vocative(self, detector: PositiveMarkerDetector) -> None:
        """Test detection of vocative case."""
        text = "Пане Андрію, як справи?"
        result = detector.detect(text)
        assert result.count >= 1
        assert any(m.category == "vocative" for m in result.matches)

    def test_detect_particle_zhe(self, detector: PositiveMarkerDetector) -> None:
        """Test detection of particle 'же'."""
        text = "Він же прийшов вчора!"
        result = detector.detect(text)
        assert result.count >= 1
        assert any(m.category == "particles" for m in result.matches)

    def test_detect_particle_bo(self, detector: PositiveMarkerDetector) -> None:
        """Test detection of particle 'бо'."""
        text = "Бо треба було раніше подумати."
        result = detector.detect(text)
        assert result.count >= 1

    def test_detect_conjunction_prote(self, detector: PositiveMarkerDetector) -> None:
        """Test detection of conjunction 'проте'."""
        text = "Проте це не означає поразки."
        result = detector.detect(text)
        assert result.count >= 1
        assert any(m.category == "conjunctions" for m in result.matches)

    def test_detect_diminutive(self, detector: PositiveMarkerDetector) -> None:
        """Test detection of diminutive forms."""
        text = "Яке гарне сонечко сьогодні!"
        result = detector.detect(text)
        assert result.count >= 1

    def test_nativeness_score(self, detector: PositiveMarkerDetector) -> None:
        """Test nativeness score calculation."""
        # Text with positive markers
        text = "Пане Іване, бо ж треба діяти! Проте, варто подумати."
        result = detector.detect(text)
        score = detector.calculate_nativeness_score(result)
        assert 0 <= score <= 100

    def test_nativeness_score_plain_text(self, detector: PositiveMarkerDetector) -> None:
        """Test nativeness score for plain text."""
        text = "Це простий текст без особливих маркерів."
        result = detector.detect(text)
        score = detector.calculate_nativeness_score(result)
        # Plain text should have lower score
        assert score < 50

    def test_category_breakdown(self, detector: PositiveMarkerDetector) -> None:
        """Test category breakdown."""
        text = "Друже, адже ми ж домовлялися! Отже, варто спробувати."
        result = detector.detect(text)
        breakdown = detector.get_category_breakdown(result)
        assert isinstance(breakdown, dict)

    def test_missing_categories(self, detector: PositiveMarkerDetector) -> None:
        """Test identifying missing categories."""
        text = "Простий текст."
        result = detector.detect(text)
        missing = detector.get_missing_categories(result)
        assert isinstance(missing, list)
        # Most categories should be missing
        assert len(missing) > 0

    def test_improvement_suggestions(self, detector: PositiveMarkerDetector) -> None:
        """Test improvement suggestions."""
        text = "Простий текст без маркерів."
        result = detector.detect(text)
        suggestions = detector.get_improvement_suggestions(result)
        assert isinstance(suggestions, list)
        if suggestions:
            assert "category" in suggestions[0]
            assert "suggestion" in suggestions[0]

    def test_analyze_balance(self, detector: PositiveMarkerDetector) -> None:
        """Test balance analysis."""
        text = "Пане Іване, бо ж треба діяти! Проте варто подумати, ось так."
        result = detector.detect(text)
        analysis = detector.analyze_balance(result)
        assert "total_markers" in analysis
        assert "diversity_score" in analysis
        assert "categories_present" in analysis


# ============================================================================
# Fertility Calculator Tests
# ============================================================================


class TestFertilityCalculator:
    """Tests for FertilityCalculator."""

    @pytest.fixture
    def calculator(self) -> FertilityCalculator:
        """Create calculator instance."""
        return FertilityCalculator()

    def test_basic_calculation(self, calculator: FertilityCalculator) -> None:
        """Test basic fertility calculation."""
        text = "Це простий текст українською мовою."
        result = calculator.calculate(text)
        assert result.word_count == 5
        assert result.token_count > 0
        assert result.fertility_rate > 0

    def test_precomputed_tokens(self, calculator: FertilityCalculator) -> None:
        """Test with precomputed token count."""
        text = "Це текст."
        result = calculator.calculate(text, precomputed_tokens=4)
        assert result.token_count == 4
        assert result.fertility_rate == 2.0  # 4 tokens / 2 words

    def test_empty_text(self, calculator: FertilityCalculator) -> None:
        """Test empty text handling."""
        result = calculator.calculate("")
        assert result.word_count == 0
        assert result.fertility_rate == 0.0

    def test_quality_level(self, calculator: FertilityCalculator) -> None:
        """Test quality level determination."""
        # Simulate different fertility rates
        text = "Текст"

        # Excellent (< 1.5)
        result = calculator.calculate(text, precomputed_tokens=1)
        assert result.quality_level == "excellent"

        # Good (1.5 - 2.0, exclusive)
        # 2 tokens / 1 word = 2.0, which is "acceptable" at boundary
        result = calculator.calculate(text, precomputed_tokens=2)
        assert result.quality_level == "acceptable"

        # Very poor (> 3.0)
        result = calculator.calculate(text, precomputed_tokens=3)
        assert result.quality_level == "very_poor"

    def test_batch_calculation(self, calculator: FertilityCalculator) -> None:
        """Test batch calculation."""
        texts = ["Перший текст.", "Другий текст тут.", "Третій."]
        results = calculator.calculate_batch(texts)
        assert len(results) == 3
        assert all(r.word_count > 0 for r in results)

    def test_aggregate_calculation(self, calculator: FertilityCalculator) -> None:
        """Test aggregate statistics."""
        texts = ["Перший текст.", "Другий текст тут.", "Третій тексточок."]
        aggregate = calculator.calculate_aggregate(texts)
        assert "total_texts" in aggregate
        assert aggregate["total_texts"] == 3
        assert "aggregate_fertility_rate" in aggregate
        assert "mean_fertility_rate" in aggregate

    def test_compare_texts(self, calculator: FertilityCalculator) -> None:
        """Test text comparison."""
        text_a = "Короткий."
        text_b = "Довший текст тут."
        comparison = calculator.compare_texts(text_a, text_b)
        assert "text_a_fertility" in comparison
        assert "text_b_fertility" in comparison
        assert "more_efficient" in comparison

    def test_is_efficient_property(self, calculator: FertilityCalculator) -> None:
        """Test is_efficient property."""
        # Efficient (< 2.0)
        result = calculator.calculate("Текст", precomputed_tokens=1)
        assert result.is_efficient

        # Not efficient (>= 2.0)
        result = calculator.calculate("Слово", precomputed_tokens=5)
        assert not result.is_efficient


class TestFertilityConvenienceFunctions:
    """Tests for fertility convenience functions."""

    def test_calculate_fertility_function(self) -> None:
        """Test calculate_fertility convenience function."""
        rate = calculate_fertility("Це простий текст.")
        assert rate > 0

    def test_evaluate_fertility_quality(self) -> None:
        """Test evaluate_fertility_quality function."""
        assert evaluate_fertility_quality(1.2) == "excellent"
        assert evaluate_fertility_quality(1.7) == "good"
        assert evaluate_fertility_quality(2.2) == "acceptable"
        assert evaluate_fertility_quality(2.7) == "poor"
        assert evaluate_fertility_quality(3.5) == "very_poor"


# ============================================================================
# Integration Tests
# ============================================================================


class TestDetectorIntegration:
    """Integration tests for detector ecosystem."""

    def test_all_detectors_load(self) -> None:
        """Test that all detectors can be loaded."""
        russism = create_russism_detector()
        anglicism = create_anglicism_detector()
        positive = create_positive_marker_detector()

        assert russism.pattern_count > 0
        assert anglicism.pattern_count > 0
        assert positive.pattern_count > 0

    def test_combined_analysis(self) -> None:
        """Test combined analysis with all detectors."""
        text = (
            "Пане Іване, треба прийняти участь у мітингу. "
            "Бо ж дедлайн завтра! Проте, варто подумати."
        )

        russism = create_russism_detector()
        anglicism = create_anglicism_detector()
        positive = create_positive_marker_detector()
        fertility = FertilityCalculator()

        russism_result = russism.detect(text)
        anglicism_result = anglicism.detect(text)
        positive_result = positive.detect(text)
        fertility_result = fertility.calculate(text)

        # Should detect russisms
        assert russism_result.count > 0

        # Should detect anglicisms
        assert anglicism_result.count > 0

        # Should detect positive markers
        assert positive_result.count > 0

        # Should calculate fertility
        assert fertility_result.fertility_rate > 0

    def test_detector_statistics_consistency(self) -> None:
        """Test that statistics are consistent across detectors."""
        detectors = [
            create_russism_detector(),
            create_anglicism_detector(),
            create_positive_marker_detector(),
        ]

        for detector in detectors:
            stats = detector.get_statistics()
            assert stats["total_patterns"] == sum(stats["by_category"].values())

    def test_overlap_removal(self) -> None:
        """Test that overlapping matches are properly removed."""
        detector = create_russism_detector()
        # Text with potential overlapping patterns
        text = "На протязі дня прийняти участь."
        result = detector.detect(text)

        # Check no overlaps
        for i, match1 in enumerate(result.matches):
            for match2 in result.matches[i + 1:]:
                # No overlap should exist
                assert match1.end <= match2.start or match2.end <= match1.start
