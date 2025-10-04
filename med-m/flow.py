import os
from crewai import Flow
from crewai.flow.flow import listen, start, and_
from tavily import TavilyClient
from src.hugging_face_ner import process_ner_output, generate_clean_ner_report
from src.crew.agents_and_taks import (
    ner_validation_crew,
    prelim_diag_crew,
    report_writing_crew,
)


class CdssPipeline(Flow):
    def __init__(self, sample_text: str, tavily_api: str):
        super().__init__()
        self.sample_text = sample_text
        self.tavily_api = tavily_api

    # -------------------------
    # Step 1: Run HuggingFace NER
    # -------------------------
    @start()
    def initial_hugging_face_ner_report(self):
        tagged_tokens, unique_tags = process_ner_output(self.sample_text)
        report = generate_clean_ner_report(tagged_tokens, unique_tags)

        return {"report": report, "sample_text": self.sample_text}

    # -------------------------
    # Step 2: Validate NER
    # -------------------------
    @listen(initial_hugging_face_ner_report)
    def ner_validation_method(self, report_dict):
        report = report_dict["report"]
        sample_text = report_dict["sample_text"]

        post_ner_report = ner_validation_crew.kickoff(
            inputs={"input_text": sample_text, "ner_output": report}
        )
        self.state["post_ner_report"] = post_ner_report
        return post_ner_report

    # -------------------------
    # Step 3: Preliminary Diagnosis
    # -------------------------
    @listen(ner_validation_method)
    def prelim_diag_method(self, post_ner_report):
        prelim_report = prelim_diag_crew.kickoff(
            inputs={"output_count": 3, "post_ner_report": post_ner_report}
        )
        self.state["prelim_report"] = prelim_report
        return prelim_report

    # -------------------------
    # Step 4: Extract only diagnosis strings
    # -------------------------
    @listen(prelim_diag_method)
    def extract_diagnosis_method(self, prelim_report):
        try:
            diagnosis = [entry.preliminary_diagnosis for entry in prelim_report["entries"]]
        except Exception:
            # fallback if prelim_report is dict or raw text
            if isinstance(prelim_report, dict) and "entries" in prelim_report:
                diagnosis = [e["preliminary_diagnosis"] for e in prelim_report["entries"]]
            else:
                diagnosis = [str(prelim_report)]
        return diagnosis

    # -------------------------
    # Step 5: Tavily Best Practices Search
    # -------------------------
    @listen(extract_diagnosis_method)
    def best_practises(self, diagnosis):
        tavily_client = TavilyClient(api_key=self.tavily_api)
        best_practices_summary = []

        for entry in diagnosis:
            response = tavily_client.search(
                f"Best Practices for {entry}", search_depth="advanced"
            )

            results = [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "summary": result.get("content", ""),
                }
                for result in response.get("results", [])
                if result.get("score", 0) > 0.5
            ]

            best_practices_summary.append(
                {"best_practices_for": entry, "practices": results}
            )

        self.state["best_practices_summary"] = best_practices_summary
        return best_practices_summary

    # -------------------------
    # Step 6: Report Writing
    # -------------------------
    @listen(and_(ner_validation_method, prelim_diag_method, best_practises))
    def report_writing_method(self):
        result = report_writing_crew.kickoff(
            inputs={
                "post_ner_report": self.state["post_ner_report"],
                "prelim_diagnosis": self.state["prelim_report"],
                "best_pracs": self.state["best_practices_summary"],
            }
        )
        return result


# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    sample_text = """42-year-old male presenting with a history of type 2 diabetes diagnosed 10 years ago.
    Current Complaint are Complains of polyuria, polydipsia, blurred vision, and fatigue for the past 2 weeks.
    Physical Examination revealed Blood pressure 140/90 mmHg. Fundoscopic examination reveals microaneurysms and cotton wool spots.
    Rest of the examination is unremarkable.
    Investigations reported Random blood sugar 350 mg/dl, HbA1c 10.5%.
    Intravenous fluids, insulin infusion, and electrolyte replacement initiated."""

    flow = CdssPipeline(sample_text=sample_text, tavily_api=os.getenv("TAVILY_API_KEY"))
    flow.plot("project_flow")  # saves flow diagram
    final_report = flow.kickoff()

    print("\n==== FINAL CLINICAL REPORT ====\n")
    print(final_report)
