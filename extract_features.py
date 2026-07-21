import os
import json
import logging
import hashlib
import time
import threading
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    open_science_practices: List[str] = Field(
        description="是否提及开源实践（如数据开源、代码开源、材料开源、研究预注册，英文简写表示，如 ['Open Data', 'Open Code', 'Preregistration']，若无提及则填 ['None']）"
    )
    statistical_reporting_style: str = Field(
        description="文章汇报数据时的统计倾向与风格描述（如 'Reported P-values and confidence intervals', 'Reported bootstrapped confidence intervals for mediation', 'None' 等）"
    )


class FeatureExtractor:
    """
    层②：结构化提取层。通过并发调用 LLM，把大样本量非结构化英文摘要转为统一规格的 Pydantic 实体字典。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.failed_papers = []

    def extract_paper(self, paper: PaperRecord) -> Optional[ExtractedFeatures]:
        """
        对单篇论文摘要调用 LLM 执行信息抽取，内置本地特征缓存层以节省 API Token 成本。
        """
        from llm_client import get_prompt_fingerprint, EXTRACTION_PROMPT_VERSION

        paper_id_clean = paper.id or paper.doi or "unknown"
        paper_hash = hashlib.md5(paper_id_clean.encode("utf-8")).hexdigest()[:12]
        
        system_prompt = "You are an expert academic metadata extractor. Output valid JSON only."
        fingerprint = get_prompt_fingerprint(
            prompt_version=EXTRACTION_PROMPT_VERSION,
            model_name=self.llm.model,
            temperature=0.1,
            system_prompt=system_prompt
        )
        
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"feature_{paper_hash}_{fingerprint}.json")

        # 检查本地缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    raw_json = json.load(f)
                raw_json["paper_id"] = paper_id_clean
                feature_obj = ExtractedFeatures.model_validate(raw_json)
                return feature_obj
            except Exception as e_cache:
                logger.warning(f"读取论文特征缓存失败: {e_cache}")

        # 使用 QPS 限速保护 API 不被频限 (频率控制：2 QPS)
        self.rate_limit_qps()

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
7. `open_science_practices` (list of strings): List of open science practices mentioned or implied (e.g., ["Open Data"], ["Open Code"], ["Preregistration"]). Use ["None"] if not mentioned.
8. `statistical_reporting_style` (string): Summary of how statistics or hypotheses are tested and reported (e.g., "Reported P-values and mediation confidence intervals", "Reported Bayesian factor analysis", "None" if qualitative).

Output MUST be clean valid JSON only without extra conversational markdown.
"""
        try:
            raw_json = self.llm.call_json(prompt=prompt)
            raw_json["paper_id"] = paper_id_clean
            feature_obj = ExtractedFeatures.model_validate(raw_json)

            # 保存至缓存
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(feature_obj.model_dump(), f, ensure_ascii=False, indent=2)
            except Exception as e_w_cache:
                logger.warning(f"写入论文特征缓存文件失败: {e_w_cache}")

            return feature_obj
        except Exception as e:
            logger.warning(f"论文 [{paper.title[:30]}...] 特征抽取出现异常: {str(e)}")
            raise e

    # 类级别的 QPS 控制机制
    _qps_lock = threading.Lock() if 'threading' in globals() else None
    _last_req_time = [0.0]

    def rate_limit_qps(self):
        """
        线程安全的 QPS 限速机制，确保请求间隔不小于 0.5s (2 QPS 限速)。
        """
        import threading
        if not FeatureExtractor._qps_lock:
            FeatureExtractor._qps_lock = threading.Lock()
        
        with FeatureExtractor._qps_lock:
            now = time.time()
            elapsed = now - FeatureExtractor._last_req_time[0]
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)
            FeatureExtractor._last_req_time[0] = time.time()

    def extract_batch(self, papers: List[PaperRecord], max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        利用多线程池并发批量解析大样本论文列表，实现百篇级高频特征极速结构化抽取
        """
        import threading
        from datetime import datetime
        logger.info(f"开始开启并发线程池 (Workers={max_workers})，批量结构化解析 {len(papers)} 篇大样本论文特征...")
        extracted_results: List[Dict[str, Any]] = []
        self.failed_papers = []

        def process_one(p: PaperRecord) -> Optional[Dict[str, Any]]:
            feat = self.extract_paper(p)
            if feat:
                feat_dict = feat.model_dump()
                feat_dict["title"] = p.title
                feat_dict["abstract"] = p.abstract
                feat_dict["cited_by_count"] = p.cited_by_count
                feat_dict["publication_year"] = p.publication_year
                feat_dict["concepts"] = getattr(p, "concepts", [])
                return feat_dict
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {executor.submit(process_one, p): p for p in papers}
            for future in tqdm(as_completed(future_to_paper), total=len(papers), desc="大样本并发抽取"):
                p = future_to_paper[future]
                try:
                    res = future.result()
                    if res:
                        extracted_results.append(res)
                    else:
                        self.failed_papers.append({
                            "paper_id": p.id or p.doi or "unknown",
                            "title": p.title,
                            "error_type": "ExtractionFailure",
                            "error_message": "LLM extraction returned null feature object",
                            "retry_count": 3,
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as exc:
                    self.failed_papers.append({
                        "paper_id": p.id or p.doi or "unknown",
                        "title": p.title,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "retry_count": 3,
                        "timestamp": datetime.now().isoformat()
                    })

        logger.info(f"并发解析完成，成功获得 {len(extracted_results)}/{len(papers)} 篇大样本有效特征实体。")
        return extracted_results

