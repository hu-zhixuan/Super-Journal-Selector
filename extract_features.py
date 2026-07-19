import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
from llm_client import LLMClient
from fetch_papers import PaperRecord

logger = logging.getLogger(__name__)


class ExtractedFeatures(BaseModel):
    """
    单篇论文摘要中由 LLM 提取的规范化特征实体
    """
    paper_id: str = Field(description="原始论文ID或DOI")
    method_category: str = Field(
        description="研究方法分类，严格在以下五类中单选其一：Quantitative_Empirical(定量实证/计量/问卷), Qualitative_CaseStudy(定性/案例/扎根/访谈), Mixed_Methods(混合研究), Theoretical_Review(纯理论/综述/观点), Computational_AI_Simulation(计算社会科学/AI/仿真建模)"
    )
    sample_description: str = Field(
        description="样本量及数据来源简述（例如 'N=1,234家上市企业面板数据', '3个深度的跨国并购案例', 'Twitter 40万条文本数据' 或 '无实证数据'）"
    )
    sample_size_approx: int = Field(
        description="估算的定量分析有效样本量总数值（如 1234, 400000），若为纯理论/定性/不可统计则填 -1"
    )
    theoretical_frameworks: List[str] = Field(
        description="该文依托或对话的核心理论框架/核心构念（最多列出3个英文标准名称，如 ['Resource-Based View', 'Self-Determination Theory'] 或基本构念）"
    )
    analytical_tools: List[str] = Field(
        description="实际采用的计量软件/统计模型/机器学习模型/分析工具（如 ['SEM-PLS', 'Difference-in-Differences', 'BERT', 'NVivo']，最多列出4个）"
    )
    novelty_highlight: str = Field(
        description="一句话总结该论文能被目标刊物接收的关键创新点（如：首次将双重差分模型应用于异质性干预机制，或者采用了极为独特的跨国微观匹配数据）"
    )


class FeatureExtractor:
    """
    层②：结构化提取层。通过 LLM 把非结构化英文摘要转为统一规格的 Pydantic 实体字典。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def extract_paper(self, paper: PaperRecord) -> Optional[ExtractedFeatures]:
        """
        对单篇论文摘要调用 LLM 执行信息抽取
        """
        prompt = f"""
You are an expert academic research reviewer. Read the following paper title and abstract carefully, then extract key methodological and theoretical features into valid JSON format matching the schema below.

Paper Title: {paper.title}
Abstract: {paper.abstract}

Required JSON Schema Fields:
1. `method_category` (string): Strictly select EXACTLY ONE of: "Quantitative_Empirical", "Qualitative_CaseStudy", "Mixed_Methods", "Theoretical_Review", "Computational_AI_Simulation".
2. `sample_description` (string): Brief description of sample scale and data source in concise text.
3. `sample_size_approx` (integer): Approximate numeric total sample size (e.g. 1500, 250). Return -1 if purely theoretical, case study without N, or not applicable.
4. `theoretical_frameworks` (list of strings): Up to 3 main theoretical lenses or core conceptual constructs.
5. `analytical_tools` (list of strings): Up to 4 statistical, econometric, computational tools or models used (e.g., OLS, DID, SEM, LLM, Regression).
6. `novelty_highlight` (string): A concise 1-sentence analytical note on what makes this research design or contribution standout for publication.

Output MUST be clean valid JSON only without extra conversational markdown.
"""
        try:
            raw_json = self.llm.call_json(prompt=prompt)
            # 将原始 paper_id 注入并校验
            raw_json["paper_id"] = paper.id or paper.doi or "unknown"
            feature_obj = ExtractedFeatures.model_validate(raw_json)
            return feature_obj
        except Exception as e:
            logger.warning(f"论文 [{paper.title[:30]}...] 特征抽取出现异常: {str(e)}")
            return None

    def extract_batch(self, papers: List[PaperRecord]) -> List[Dict[str, Any]]:
        """
        批量解析论文列表并汇总输出结构化数据
        """
        logger.info(f"开始使用 LLM 批量结构化解析 {len(papers)} 篇论文特征...")
        extracted_results: List[Dict[str, Any]] = []

        for paper in tqdm(papers, desc="提取论文特征"):
            feat = self.extract_paper(paper)
            if feat:
                feat_dict = feat.model_dump()
                # 附加上原始论文参数以便后续聚合与高引展示
                feat_dict["title"] = paper.title
                feat_dict["cited_by_count"] = paper.cited_by_count
                feat_dict["publication_year"] = paper.publication_year
                extracted_results.append(feat_dict)

        logger.info(f"结构化解析完成，成功获得 {len(extracted_results)}/{len(papers)} 篇文献特征。")
        return extracted_results
