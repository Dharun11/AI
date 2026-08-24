import mimetypes
from datetime import UTC, datetime
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel

from ingestion.hashing import sha256_bytes
from ingestion.models import Document, Element, ElementType, TableData
from ingestion.parsers.factory import register_parser

_HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}

_LABEL_TO_ELEMENT_TYPE = {
    DocItemLabel.TEXT: ElementType.PARAGRAPH,
    DocItemLabel.PARAGRAPH: ElementType.PARAGRAPH,
    DocItemLabel.LIST_ITEM: ElementType.LIST,
    DocItemLabel.CODE: ElementType.CODE,
    DocItemLabel.CAPTION: ElementType.CAPTION,
}


@register_parser("docling")
class DoclingParser:
    """Parses PDF/DOCX/PPTX/HTML/Markdown into Document + Element via Docling."""

    name = "docling"

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def parse(self, source: Path) -> tuple[Document, list[Element]]:
        raw_bytes = source.read_bytes()
        content_hash = sha256_bytes(raw_bytes)
        mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"

        docling_doc = self._converter.convert(source).document

        document = Document(
            doc_id=content_hash,
            source_uri=source.resolve().as_uri(),
            mime_type=mime_type,
            content_hash=content_hash,
            ingested_at=datetime.now(UTC),
            doc_metadata={"title": docling_doc.name, "parser_backend": self.name},
        )

        elements = list(self._iter_elements(docling_doc, document.doc_id))
        return document, elements

    def _iter_elements(self, docling_doc, doc_id: str):
        section_stack: dict[int, str] = {}
        order_index = 0

        for item, _ in docling_doc.iterate_items():
            label = getattr(item, "label", None)
            table_data = None

            if label in _HEADING_LABELS:
                heading_level = getattr(item, "level", 0) or 0
                section_path = [
                    section_stack[lvl] for lvl in sorted(section_stack) if lvl < heading_level
                ]
                section_stack = {
                    lvl: text for lvl, text in section_stack.items() if lvl < heading_level
                }
                section_stack[heading_level] = item.text
                element_type = ElementType.HEADING
                text = item.text
            elif label == DocItemLabel.TABLE:
                element_type = ElementType.TABLE
                table_data = self._build_table_data(item, docling_doc)
                text = table_data.markdown
                section_path = [section_stack[lvl] for lvl in sorted(section_stack)]
            elif label in _LABEL_TO_ELEMENT_TYPE:
                element_type = _LABEL_TO_ELEMENT_TYPE[label]
                text = item.text
                section_path = [section_stack[lvl] for lvl in sorted(section_stack)]
            else:
                continue

            page_no = item.prov[0].page_no if getattr(item, "prov", None) else None
            yield Element(
                element_id=f"{doc_id}-el-{order_index}",
                doc_id=doc_id,
                order_index=order_index,
                type=element_type,
                text=text,
                table_data=table_data,
                page_range=(page_no, page_no) if page_no is not None else None,
                section_path=section_path,
            )
            order_index += 1

    @staticmethod
    def _build_table_data(table_item, docling_doc) -> TableData:
        header_rows: list[list[str]] = []
        body_rows: list[list[str]] = []
        for row in table_item.data.grid:
            row_text = [cell.text for cell in row]
            if row and all(cell.column_header for cell in row):
                header_rows.append(row_text)
            else:
                body_rows.append(row_text)

        return TableData(
            header_rows=header_rows,
            body_rows=body_rows,
            caption=table_item.caption_text(docling_doc) or None,
            markdown=table_item.export_to_markdown(docling_doc),
        )
