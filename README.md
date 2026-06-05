# AI IEEE Crawler and Paper Writer

This repository contains an AI-assisted workflow for crawling IEEE papers and generating academic paper drafts. It includes two workflows:

- `DownloadPDF`: search and download IEEE papers, then convert PDFs with MinerU.
- `Analysis`: clean local paper text, build a retrieval index, and generate academic report sections with LLMs.

## Repository Layout

```text
.
├── Analysis/       # RAG-based report generation workflow
├── DownloadPDF/    # IEEE paper collection and MinerU conversion workflow
└── environment.yml # Conda environment
```

Generated files are intentionally excluded from Git, including raw papers, PDFs, browser profiles, vector indexes, intermediate outputs, `.env` files, and Python caches.

## Setup

```bash
conda env create -f environment.yml
conda activate env
```

Create environment files from the examples:

```bash
cp .env.example .env
cp Analysis/.env.example Analysis/.env
cp DownloadPDF/.env.example DownloadPDF/.env
```

Fill in the required keys:

- `QWEN_API_KEY`: embedding API key used by `Analysis`
- `MINIMAX_API_KEY`: LLM API key used by `Analysis`
- `MINERU_TOKEN`: MinerU API token used by `DownloadPDF/mineru_convert.py`

## Run

Download and convert papers:

```bash
cd DownloadPDF
python ieee_pipeline.py
python mineru_convert.py
```

Generate the report:

```bash
cd ../Analysis
python -m src.main
python -m src.polish_standard_paper
```

To change the paper topic, edit:

```text
Analysis/configs/topic.yaml
```

## Notes

- IEEE login may require manual browser interaction.
- Keep API keys in `.env` files only.
- Before publishing derived datasets or papers, verify that you have the right to redistribute them.

## License

MIT
