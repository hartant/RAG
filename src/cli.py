"""Command line interface of the RAG pipeline."""

from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .indexer import build_index, collect_files, load_index, save_index
from .models import MinimalSearchResults, RagDataset, StudentSearchResults
from .evaluator import recall_at_k
from .models import AnsweredQuestion
from .retriever import Retriever
from .generator import Generator, build_context
from .models import MinimalAnswer, StudentSearchResultsAndAnswer

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SEARCH_DIR = "data/output/search_results"


class Cli:
    """Commands exposed through Python Fire."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = RAW_DIR,
        index_dir: str = PROCESSED_DIR,
    ) -> None:
        """Chunk every file under raw_dir and save the index."""
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("Error: max_chunk_size must be between 1 and 2000.")
            return
        paths = collect_files(raw_dir)
        if not paths:
            print(f"Error: no .py or .md files found in {raw_dir}.")
            return
        try:
            chunks, postings, lengths = build_index(paths, max_chunk_size)
            save_index(index_dir, chunks, postings, lengths)
        except OSError as exc:
            print(f"Error: cannot write the index: {exc}")
            return
        print(
            "Ingestion complete! "
            f"Indexed {len(chunks)} chunks under {index_dir}/"
        )

    def _load(self, index_dir: str) -> Optional[Retriever]:
        """Load the index and build a Retriever, or None on failure."""
        try:
            chunks, postings, lengths = load_index(index_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(
                f"Error: cannot load the index ({exc}). "
                "Run the index command first."
            )
            return None
        return Retriever(chunks, postings, lengths)

    def search(
        self,
        query: str,
        k: int = 10,
        index_dir: str = PROCESSED_DIR,
    ) -> None:
        """Print the top-k sources for one query."""
        if not str(query).strip():
            print("Error: the query is empty.")
            return
        if k <= 0:
            print("Error: k must be greater than 0.")
            return
        retriever = self._load(index_dir)
        if retriever is None:
            return
        for src in retriever.search(str(query), k):
            print(
                f"{src.file_path} "
                f"[{src.first_character_index}:"
                f"{src.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = SEARCH_DIR,
        index_dir: str = PROCESSED_DIR,
    ) -> None:
        """Run the search on every question of a dataset."""
        if k <= 0:
            print("Error: k must be greater than 0.")
            return
        try:
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except OSError as exc:
            print(f"Error: cannot read {dataset_path}: {exc}")
            return
        except ValueError as exc:
            print(f"Error: invalid dataset file: {exc}")
            return
        retriever = self._load(index_dir)
        if retriever is None:
            return
        results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching"):
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=retriever.search(q.question, k),
                )
            )
        output = StudentSearchResults(search_results=results, k=k)
        out_path = Path(save_directory) / Path(dataset_path).name
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Error: cannot write {out_path}: {exc}")
            return
        print(f"Saved student_search_results to {out_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        """Print recall@1/3/5/10 against a ground-truth dataset."""
        try:
            with open(student_search_results_path, encoding="utf-8") as f:
                student = StudentSearchResults.model_validate_json(f.read())
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except OSError as exc:
            print(f"Error: cannot read file: {exc}")
            return
        except ValueError as exc:
            print(f"Error: invalid file: {exc}")
            return
        truth = {
            q.question_id: q
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion)
        }
        for kk in (1, 3, 5, 10):
            if kk <= student.k:
                score = recall_at_k(student, truth, kk)
                print(f"Recall@{kk}: {score:.3f} ({score * 100:.1f}%)")

    def answer(
        self,
        query: str,
        k: int = 10,
        index_dir: str = PROCESSED_DIR,
    ) -> None:
        """Retrieve context for one query and print a grounded answer."""
        if not str(query).strip():
            print("Error: the query is empty.")
            return
        if k <= 0:
            print("Error: k must be greater than 0.")
            return
        retriever = self._load(index_dir)
        if retriever is None:
            return
        sources = retriever.search(str(query), k)
        context = build_context(sources)
        try:
            generator = Generator()
            print(generator.answer(str(query), context))
        except Exception as exc:  # model load/generation failure
            print(f"Error: cannot generate an answer: {exc}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate an answer for every question of a search result."""
        try:
            with open(student_search_results_path, encoding="utf-8") as f:
                student = StudentSearchResults.model_validate_json(f.read())
        except OSError as exc:
            print(
                f"Error: cannot read {student_search_results_path}: {exc}"
            )
            return
        except ValueError as exc:
            print(f"Error: invalid search results file: {exc}")
            return
        try:
            generator = Generator()
        except Exception as exc:
            print(f"Error: cannot load the model: {exc}")
            return
        answers: List[MinimalAnswer] = []
        for res in tqdm(student.search_results, desc="Answering"):
            context = build_context(res.retrieved_sources)
            try:
                text = generator.answer(res.question, context)
            except Exception as exc:
                text = f"Error while generating the answer: {exc}"
            answers.append(
                MinimalAnswer(
                    question_id=res.question_id,
                    question=res.question,
                    retrieved_sources=res.retrieved_sources,
                    answer=text,
                )
            )
        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=student.k,
        )
        out_path = (
            Path(save_directory) /
            Path(student_search_results_path).name
        )
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                output.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Error: cannot write {out_path}: {exc}")
            return
        print(f"Saved student_search_results_and_answer to {out_path}")
