"""PDF reader tool using pypdf."""

from pathlib import Path

from pypdf import PdfReader

MAX_OUTPUT_CHARS = 50000


def read_pdf(file_path: str) -> str:
    """
    Read and extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text content (truncated to MAX_OUTPUT_CHARS).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF or cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    try:
        reader = PdfReader(path)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e!s}") from e

    if len(reader.pages) == 0:
        return "PDF contains no pages."

    text_parts: list[str] = []
    total_chars = 0

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
            total_chars += len(text)
            if total_chars >= MAX_OUTPUT_CHARS:
                break

    full_text = "\n\n".join(text_parts)

    if total_chars > MAX_OUTPUT_CHARS:
        full_text = full_text[:MAX_OUTPUT_CHARS] + f"\n\n[Truncated at {MAX_OUTPUT_CHARS} characters]"

    return full_text
