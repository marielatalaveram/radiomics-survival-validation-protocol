# shared setup, sourced by the other scripts
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(here)
})

# RADIOMICS_FAST=1 -> smoke-test folders, same as in the python code
fast        <- identical(Sys.getenv("RADIOMICS_FAST"), "1")
results_dir <- here(if (fast) "results_smoke" else "results")
fig_dir     <- here(if (fast) "figures_smoke" else "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(results_dir, "tables"), showWarnings = FALSE, recursive = TRUE)

read_result <- function(...) {
  path <- file.path(results_dir, ...)
  if (!file.exists(path)) stop("Missing ", path, ". Run the notebook first.", call. = FALSE)
  # tables use ";" (see CSV_SEP in config.py); older comma files are still read
  delim <- if (grepl(";", readLines(path, n = 1, warn = FALSE))) ";" else ","
  readr::read_delim(path, delim = delim, show_col_types = FALSE)
}

save_fig <- function(plot, name, width, height, formats = c("pdf"), dpi = 600, ...) {
  for (ext in formats) {
    ggsave(file.path(fig_dir, paste0(name, ".", ext)), plot, width = width, height = height,
           dpi = dpi, ...)
  }
}
