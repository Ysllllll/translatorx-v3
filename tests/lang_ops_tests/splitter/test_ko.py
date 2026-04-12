"""Korean (ko) splitter tests."""

from lang_ops import TextOps, ChunkPipeline
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses
from ._base import SplitterTestBase


TEXT_SAMPLE: str = '한국의 대중문화는 최근 수십 년간 전 세계적으로 놀라운 성장을 이루었다. 음악, 영화, 드라마 등 다양한 분야에서 한국 콘텐츠가 큰 인기를 끌고 있다；특히 K-pop은 남미와 유럽에서도 열렬한 팬덤을 형성했다. 한류 현상은 단순한 문화 수출을 넘어, 국가 이미지 제고와 경제적 효과까지 가져왔다! 그러나 이러한 성공 뒤에는 수많은 노력과 혁신이 숨어 있다. 한국의 엔터테인먼트 산업은 철저한 기획과 시스템적인 관리로 세계적인 경쟁력을 확보했다；또한, 디지털 기술을 적극적으로 활용하여 글로벌 시장에 빠르게 진출할 수 있었다. 플랫폼 경제의 발전은 콘텐츠 소비 방식을 근본적으로 변화시켰다……이제 누구나 스마트폰 하나로 세계 각국의 콘텐츠를 즐길 수 있다. 하지만 이러한 변화는 모든 사람에게 긍정적인 것일까? 아니면 문화의 획일화라는 부작용도 존재하는 것일까? 이 질문에 대한 답은 결코 간단하지 않다. 한 문화평론가는 다음과 같이 말했다. "전통과 현대의 조화가 한국 문화의 가장 큰 강점이다." 이러한 관점에서 볼 때, 한국은 과거의 유산을 보존하면서도 미래지향적인 문화를 창출하는 데 성공한 국가라 할 수 있다.'

PARAGRAPH_TEXT: str = '한국의 대중문화가 세계적으로 성장했다.\n\nK-pop은 전 세계적인 팬덤을 형성했다.\n\n전통과 현대의 조화가 중요하다.'

_ops = TextOps.for_language("ko")


class TestKoreanSplitter(SplitterTestBase):
    LANGUAGE = "ko"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        # 기본 문장 분할
        assert _ops.split_sentences("안녕하세요. 반갑습니다!") == ["안녕하세요.", " 반갑습니다!"]
        assert _ops.split_sentences("와! 정말? 네.") == ["와!", " 정말?", " 네."]

        # 연속 종결 부호
        assert _ops.split_sentences("대단해!! 정말???") == ["대단해!!", " 정말???"]
        assert _ops.split_sentences("뭐?! 네.") == ["뭐?!", " 네."]

        # 한국어는 공백 보존 (eojeol boundaries)
        assert _ops.split_sentences("이것은 테스트입니다. 다음 문장입니다.") == ["이것은 테스트입니다.", " 다음 문장입니다."]

        # 닫는 인용부호
        assert _ops.split_sentences('"안녕!" 그가 말했다.') == ['"안녕!"', ' 그가 말했다.']

        # 이모지
        assert _ops.split_sentences("대단해😊! 정말👋?") == ["대단해😊!", " 정말👋?"]

        # 빈 문자열 / 엣지 케이스
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("마침표없음") == ["마침표없음"]

    def test_split_clauses(self) -> None:
        # 기본 절 분할
        assert _ops.split_clauses("첫째, 둘째, 셋째.") == ["첫째,", " 둘째,", " 셋째."]

        # 세미콜론
        assert _ops.split_clauses("안녕하세요, 세계입니다.") == ["안녕하세요,", " 세계입니다."]

        # 연속 구분자
        assert _ops.split_clauses(",,,") == [",,,"]
        assert _ops.split_clauses("안녕,,, 세계") == ["안녕,,,", " 세계"]

        # 빈 문자열 / 엣지 케이스
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("구분자없음") == ["구분자없음"]

    def test_split_by_length(self) -> None:
        # 문자 수 기준 분할 (oversized tokens kept whole)
        assert _ops.split_by_length("가나다라마바사", max_length=3) == ["가나다라", "마바사"]

        # 피트 / 빈 문자열
        assert _ops.split_by_length("테스트", max_length=10) == ["테스트"]
        assert _ops.split_by_length("", max_length=10) == []

        # 에러
        import pytest
        with pytest.raises(ValueError):
            _ops.split_by_length("테스트", max_length=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("테스트", max_length=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("테스트", max_length=5, unit="sentence")

        # 청크 체인
        assert _ops.chunk("안녕하세요. 반갑습니다!").sentences().by_length(8).result() == [
            "안녕하세요.", "반갑습니다!",
        ]
        assert _ops.chunk("첫째, 둘째, 셋째.").clauses().by_length(5).result() == [
            "첫째,", "둘째,", "셋째.",
        ]

    def test_split_long_text(self) -> None:
        # long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            '한국의 대중문화는 최근 수십 년간 전 세계적으로 놀라운 성장을 이루었다.',
            ' 음악, 영화, 드라마 등 다양한 분야에서 한국 콘텐츠가 큰 인기를 끌고 있다；특히 K-pop은 남미와 유럽에서도 열렬한 팬덤을 형성했다.',
            ' 한류 현상은 단순한 문화 수출을 넘어, 국가 이미지 제고와 경제적 효과까지 가져왔다!',
            ' 그러나 이러한 성공 뒤에는 수많은 노력과 혁신이 숨어 있다.',
            ' 한국의 엔터테인먼트 산업은 철저한 기획과 시스템적인 관리로 세계적인 경쟁력을 확보했다；또한, 디지털 기술을 적극적으로 활용하여 글로벌 시장에 빠르게 진출할 수 있었다.',
            ' 플랫폼 경제의 발전은 콘텐츠 소비 방식을 근본적으로 변화시켰다……이제 누구나 스마트폰 하나로 세계 각국의 콘텐츠를 즐길 수 있다.',
            ' 하지만 이러한 변화는 모든 사람에게 긍정적인 것일까?',
            ' 아니면 문화의 획일화라는 부작용도 존재하는 것일까?',
            ' 이 질문에 대한 답은 결코 간단하지 않다.',
            ' 한 문화평론가는 다음과 같이 말했다.',
            ' "전통과 현대의 조화가 한국 문화의 가장 큰 강점이다."',
            ' 이러한 관점에서 볼 때, 한국은 과거의 유산을 보존하면서도 미래지향적인 문화를 창출하는 데 성공한 국가라 할 수 있다.',
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            '한국의 대중문화는 최근 수십 년간 전 세계적으로 놀라운 성장을 이루었다.',
            ' 음악,',
            ' 영화,',
            ' 드라마 등 다양한 분야에서 한국 콘텐츠가 큰 인기를 끌고 있다；',
            '특히 K-pop은 남미와 유럽에서도 열렬한 팬덤을 형성했다.',
            ' 한류 현상은 단순한 문화 수출을 넘어,',
            ' 국가 이미지 제고와 경제적 효과까지 가져왔다!',
            ' 그러나 이러한 성공 뒤에는 수많은 노력과 혁신이 숨어 있다.',
            ' 한국의 엔터테인먼트 산업은 철저한 기획과 시스템적인 관리로 세계적인 경쟁력을 확보했다；',
            '또한,',
            ' 디지털 기술을 적극적으로 활용하여 글로벌 시장에 빠르게 진출할 수 있었다.',
            ' 플랫폼 경제의 발전은 콘텐츠 소비 방식을 근본적으로 변화시켰다……이제 누구나 스마트폰 하나로 세계 각국의 콘텐츠를 즐길 수 있다.',
            ' 하지만 이러한 변화는 모든 사람에게 긍정적인 것일까?',
            ' 아니면 문화의 획일화라는 부작용도 존재하는 것일까?',
            ' 이 질문에 대한 답은 결코 간단하지 않다.',
            ' 한 문화평론가는 다음과 같이 말했다.',
            ' "전통과 현대의 조화가 한국 문화의 가장 큰 강점이다."',
            ' 이러한 관점에서 볼 때,',
            ' 한국은 과거의 유산을 보존하면서도 미래지향적인 문화를 창출하는 데 성공한 국가라 할 수 있다.',
        ]

        # long text chunk chain equivalence
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)

        # long text pipeline_paragraphs_sentences()
        assert ChunkPipeline(self.PARAGRAPH_TEXT, language=self.LANGUAGE).paragraphs().sentences().result() == [
            '한국의 대중문화가 세계적으로 성장했다.',
            'K-pop은 전 세계적인 팬덤을 형성했다.',
            '전통과 현대의 조화가 중요하다.',
        ]
