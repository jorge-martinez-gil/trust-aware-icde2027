# Information Sciences Submission Checklist

Target journal: *Information Sciences* (Elsevier), ISSN 0020-0255.

## Completed in this folder

- `main.tex` converted from IEEE conference style to Elsevier `elsarticle`
  front matter for journal submission.
- Abstract shortened to comply with the journal's 250-word maximum.
- Keywords reduced to seven indexing terms.
- Separate `highlights_information_sciences.txt` file added with five highlights,
  each under 85 characters.
- CRediT, competing-interest, funding, data/code availability, generative-AI,
  and acknowledgments sections added before the references.
- Cover letter drafted in `cover_letter_information_sciences.md`.

## Author confirmation needed before upload

- Confirm the affiliation/postal address. The manuscript currently uses:
  Software Competence Center Hagenberg GmbH (SCCH), 4232 Hagenberg im Muehlkreis,
  Austria.
- Confirm the funding statement. It currently states that no specific grant
  supported the research.
- Complete Elsevier's declaration-of-competing-interests form and upload the
  generated `.docx` file if Editorial Manager asks for it.
- Archive a tagged artifact release in a durable repository such as Zenodo, then
  update `Data and code availability` with the DOI.
- Add any ORCID, phone number, or full postal details required by Editorial
  Manager but not normally included in the manuscript PDF.
- Build the final PDF in an environment with `pdflatex`, `bibtex`, and
  `elsarticle` installed; the current Windows shell used for this preparation
  did not expose `pdflatex` on `PATH`.

## Upload set

- `main.tex`
- `references.bib`
- `figures/*.pdf`
- `tables/*.tex`
- `highlights_information_sciences.txt`
- `cover_letter_information_sciences.md`
- The generated manuscript PDF
- Any required declaration-of-interest `.docx` from Elsevier's declarations tool
