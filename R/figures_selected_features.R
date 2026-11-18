# Figure 4: forest plot (a) and Spearman matrix (b) of the features selected in the
# reference single split. Inputs: results/single_split/rad_sig_cox.csv and
# rad_sig_spearman.csv (the matrix is exported instead of the raw features, so no
# patient-level data end up in the repo). Run: Rscript R/figures_selected_features.R
source(here::here("R", "00_setup.R"))
suppressPackageStartupMessages({
  library(ggforce)      # facet_col()
  library(ggcorrplot)
  library(scales)
  library(grid)
})

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
rad_sig2 <- read_result("single_split", "rad_sig_cox.csv")

sp <- suppressMessages(read_result("single_split", "rad_sig_spearman.csv"))
matriz_spearman_raw <- as.matrix(sp[, -1])
rownames(matriz_spearman_raw) <- sp[[1]]

# ---------------------------------------------------------------------------
# 2. Feature metadata: compartment, family and short name, from the feature names
#    (Corr_T1_<family>_<feature>_seg<k>)
# ---------------------------------------------------------------------------
# seg1/seg2 never appeared among the selected features, unmapped labels are kept as they are
compartment_map <- c(seg3 = "Intramuscular fat", seg4 = "Skeletal muscle", seg5 = "Subcutaneous fat")
compartment_levels <- c("Subcutaneous fat", "Skeletal muscle", "Intramuscular fat")

resultados_df_clean <- rad_sig2 %>%
  mutate(
    seg_num = str_extract(variable, "seg\\d+"),
    Compartment = coalesce(unname(compartment_map[seg_num]), seg_num),
    Compartment = factor(Compartment,
                         levels = c(compartment_levels, setdiff(unique(Compartment), compartment_levels))),

    Feature_family = str_extract(variable, "(?<=Corr_T1_)[a-z0-9]+"),
    Feature_family = case_when(
      Feature_family == "shape" ~ "Shape",
      Feature_family == "gldm"  ~ "GLDM",
      Feature_family == "glszm" ~ "GLSZM",
      Feature_family == "ngtdm" ~ "NGTDM",
      Feature_family == "glcm"  ~ "GLCM",
      Feature_family == "glrlm" ~ "GLRLM",
      Feature_family == "firstorder" ~ "First order",
      TRUE ~ Feature_family
    ),

    Feature_name_raw = str_remove(variable, "^Corr_T1_[a-z0-9]+_"),
    Feature_name_raw = str_remove(Feature_name_raw, "_seg\\d+$"),

    Feature_name = case_when(
      Feature_name_raw == "GrayLevelNonUniformity" ~ "GLNU",
      Feature_name_raw == "MajorAxisLength"        ~ "Major Axis Length",
      Feature_name_raw %in% c("Idmn", "idmn")      ~ "Idmn",
      TRUE ~ Feature_name_raw
    ),
    Variable_Clean = Feature_name
  ) %>%
  select(-seg_num, -Feature_name_raw)

resultados_df_clean %>%
  select(variable, Compartment, Feature_family, Feature_name, HR, HR_lower95, HR_upper95) %>%
  print(n = Inf)

# ---------------------------------------------------------------------------
# 3. Shared palettes
# ---------------------------------------------------------------------------
okabe_ito <- c("#E69F00", "#DDA0DD", "#9ACD32", "#D55E00",
               "#EEAEEE", "#CC79A7", "#F0E442", "#000000")

familias_unicas <- sort(unique(na.omit(resultados_df_clean$Feature_family)))
family_palette  <- setNames(rep_len(okabe_ito, length(familias_unicas)), familias_unicas)

# ---------------------------------------------------------------------------
# 4. Panel (a) - forest plot
# ---------------------------------------------------------------------------
p_forest <- ggplot(resultados_df_clean,
                   aes(y = reorder(Variable_Clean, HR), x = HR, color = Feature_family)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey40", linewidth = 0.6) +
  geom_errorbarh(aes(xmin = HR_lower95, xmax = HR_upper95), height = 0.25, linewidth = 1) +
  geom_point(shape = 18, size = 6) +
  scale_x_log10(
    breaks = scales::breaks_log(n = 6),
    labels = scales::label_number(accuracy = 0.01),
    expand = expansion(mult = c(0.05, 0.55))
  ) +
  scale_y_discrete(expand = expansion(add = c(0.6, 0.6))) +
  coord_cartesian(xlim = c(0.05, 2), clip = "off") +
  facet_col(~ Compartment, scales = "free_y", space = "free") +
  scale_color_manual(values = family_palette, name = "Feature family") +
  labs(x = NULL, y = NULL) +
  guides(color = guide_legend(override.aes = list(size = 4, linewidth = 1))) +
  theme_bw(base_size = 16) +
  theme(
    axis.title.x = element_text(size = 19, margin = margin(t = 10)),
    axis.text.x  = element_text(size = 19),
    axis.text.y  = element_text(size = 20, face = "bold"),
    strip.background = element_rect(fill = "grey95", color = "grey30", linewidth = 0.4),
    strip.text = element_text(size = 18, face = "bold", color = "grey10"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(color = "grey88", linewidth = 0.3),
    panel.border = element_rect(color = "grey40", linewidth = 0.4),
    panel.spacing = unit(1, "lines"),
    legend.position = "none",
    plot.margin = margin(t = 10, r = 20, b = 10, l = 10)
  )

save_fig(p_forest, "fig4a_forest_plot", width = 9, height = 1.6 + 0.9 * nrow(resultados_df_clean))

# ---------------------------------------------------------------------------
# 5. Panel (b) - Spearman correlation matrix
# ---------------------------------------------------------------------------
nombres_originales <- colnames(matriz_spearman_raw)
indices <- match(nombres_originales, resultados_df_clean$variable)
stopifnot(!anyNA(indices))

meta_vars <- data.frame(
  var_original   = nombres_originales,
  Feature_name   = resultados_df_clean$Feature_name[indices],
  Feature_family = resultados_df_clean$Feature_family[indices],
  Compartment    = as.character(resultados_df_clean$Compartment[indices]),
  stringsAsFactors = FALSE
)

# Short compartment names for the axis labels
meta_vars <- meta_vars %>%
  mutate(Compartment = case_when(
    Compartment == "Visceral fat"      ~ "Visceral",
    Compartment == "Intramuscular fat" ~ "Intramuscular",
    Compartment == "Skeletal muscle"   ~ "Muscle",
    Compartment == "Subcutaneous fat"  ~ "Subcutaneous",
    TRUE ~ Compartment
  ))

nuevo_compartment_levels <- c("Intramuscular", "Muscle", "Subcutaneous", "Visceral")
meta_vars$Compartment <- factor(
  meta_vars$Compartment,
  levels = c(nuevo_compartment_levels, setdiff(unique(meta_vars$Compartment), nuevo_compartment_levels)))

meta_vars$Label_axis <- make.unique(paste0(meta_vars$Feature_name, " (",
                                           as.character(meta_vars$Compartment), ")"))

meta_vars <- meta_vars %>% arrange(Compartment, Feature_family, Feature_name)

var_order <- meta_vars$var_original
matriz_spearman <- matriz_spearman_raw[var_order, var_order, drop = FALSE]
dimnames(matriz_spearman) <- list(meta_vars$Label_axis, meta_vars$Label_axis)

p_base <- ggcorrplot(
  matriz_spearman,
  type = "lower",
  lab = 1,
  lab_size = 22,
  colors = c("#00008B", "#FFFFF0", "#FF4040"),
  title = "",
  hc.order = FALSE,
  ggtheme = theme_minimal(),
  show.diag = TRUE
)

orden_final_y <- levels(p_base$data$Var1)

lookup_idx <- match(orden_final_y, meta_vars$Label_axis)
familia_en_orden       <- meta_vars$Feature_family[lookup_idx]
compartimento_en_orden <- meta_vars$Compartment[lookup_idx]

n <- length(orden_final_y)
cambios_compartimento <- which(as.character(compartimento_en_orden[-1]) !=
                                 as.character(compartimento_en_orden[-n]))

# Panel 1: labels only
labels_df <- data.frame(
  label = factor(orden_final_y, levels = orden_final_y),
  text  = orden_final_y
)

size_mm <- 70 / 2.845276        # label size (mm)

p_labels <- ggplot(labels_df, aes(x = 1, y = label)) +
  geom_text(aes(label = text), hjust = 1, size = size_mm,
            fontface = "bold", color = "grey20") +
  scale_y_discrete(limits = orden_final_y, expand = expansion(add = 0.5)) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void() +
  theme(plot.margin = margin(0, 15, 10, 0))

# Panel 2: strip of dots coloured by feature family
dots_df <- data.frame(
  label  = factor(orden_final_y, levels = orden_final_y),
  family = familia_en_orden
)

p_dots <- ggplot(dots_df, aes(x = 1, y = label, color = family)) +
  geom_point(size = 20) +
  scale_color_manual(values = family_palette, guide = "none") +
  scale_y_discrete(limits = orden_final_y, expand = expansion(add = 0.5)) +
  scale_x_continuous(limits = c(0.9, 1.1), expand = c(0, 0)) +
  theme_void() +
  theme(plot.margin = margin(0, 15, 10, 0))

# Panel 3: the matrix
p_matrix <- p_base +
  labs(title = NULL, x = NULL) +
  guides(fill = guide_colorbar(
    title = expression("Spearman's" ~ italic(rho)),
    title.hjust = 0.6,
    barwidth = unit(5, "cm"),
    barheight = unit(20, "cm")
  )) +
  geom_hline(yintercept = cambios_compartimento + 0.5, color = "white", linewidth = 1.4) +
  theme(
    plot.title = element_blank(),
    axis.title.x = element_text(size = 18, margin = margin(t = 17)),
    axis.text.y  = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x  = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 70, face = "bold", margin = margin(b = 25)),
    legend.text  = element_text(size = 68),
    legend.box.margin = margin(0, 20, 30, 20),
    legend.key.height = unit(8, "lines"),
    plot.margin = margin(0, 1.5, 0, 0)
  )

# width of the longest label at that font size
label_font_pt <- size_mm * 2.845276
pdf(NULL)
label_widths_mm <- vapply(labels_df$text, function(txt) {
  grid::convertWidth(
    grid::grobWidth(grid::textGrob(txt, gp = grid::gpar(fontsize = label_font_pt, fontface = "bold"))),
    "mm", valueOnly = TRUE
  )
}, numeric(1))

labels_panel_width_mm <- max(label_widths_mm) + 5   # +5 mm safety margin
dots_panel_width_mm   <- 15

p_matrix_completo <- p_labels + p_dots + p_matrix +
  plot_layout(widths = grid::unit.c(
    unit(labels_panel_width_mm, "mm"),
    unit(dots_panel_width_mm, "mm"),
    unit(1, "null")
  ))

# ggsave(file.path(fig_dir, "fig4b_correlation_matrix_only.pdf"), p_matrix_completo,
#        width = 31, height = 22, bg = "white")

# ---------------------------------------------------------------------------
# 6. Legend (feature families), measured with the same device
# ---------------------------------------------------------------------------
title_txt  <- "Feature family:"
title_size <- 80
label_size <- 78
point_size_aes <- 18

gap_after_title_mm   <- 10
gap_point_text_mm    <- 6
gap_between_items_mm <- 16
point_diam_mm        <- 9

measure_width_mm <- function(text, fontsize, fontface = "plain") {
  grid::convertWidth(
    grid::grobWidth(grid::textGrob(text, gp = grid::gpar(fontsize = fontsize, fontface = fontface))),
    "mm", valueOnly = TRUE
  )
}
familias_df <- data.frame(
  family = names(family_palette),
  color  = as.character(family_palette),
  stringsAsFactors = FALSE
)

title_w      <- measure_width_mm(title_txt, title_size, "bold")
label_widths <- vapply(familias_df$family, measure_width_mm, numeric(1), fontsize = label_size)

x_point <- numeric(nrow(familias_df))
x_label <- numeric(nrow(familias_df))
cursor  <- title_w + gap_after_title_mm

for (i in seq_len(nrow(familias_df))) {
  x_point[i] <- cursor + point_diam_mm / 2
  x_label[i] <- cursor + point_diam_mm + gap_point_text_mm
  cursor     <- x_label[i] + label_widths[i] + gap_between_items_mm
}

familias_df$x_point <- x_point
familias_df$x_label <- x_label
total_width_mm <- cursor + 5
invisible(dev.off())

p_legend <- ggplot(familias_df) +
  annotate("text", x = 0, y = 1, label = title_txt, hjust = 0,
           size = title_size / 2.845276, fontface = "bold", color = "black") +
  geom_point(aes(x = x_point, y = 1, color = color), size = point_size_aes, shape = 19) +
  scale_color_identity() +
  geom_text(aes(x = x_label, y = 1, label = family), hjust = 0,
            size = label_size / 2.845276, color = "grey20") +
  scale_x_continuous(limits = c(0, total_width_mm), expand = c(0, 0)) +
  ylim(0.5, 1.5) +
  coord_cartesian(clip = "off") +
  theme_void() +
  theme(plot.margin = margin(0, 0, 0, 0))

# ---------------------------------------------------------------------------
# 7. Final composition: legend fixed at the top-left (absolute height), matrix below
# ---------------------------------------------------------------------------
top_row_height_mm <- 80
left_margin_mm    <- 5

top_panel <- plot_spacer() | wrap_elements(p_legend, clip = FALSE) | plot_spacer()
top_panel <- top_panel +
  plot_layout(widths = grid::unit.c(unit(left_margin_mm, "mm"),
                                    unit(total_width_mm, "mm"),
                                    unit(1, "null")))

panel_final <- top_panel / p_matrix_completo +
  plot_layout(heights = grid::unit.c(unit(top_row_height_mm, "mm"),
                                     unit(1, "null")))

ggsave(file.path(fig_dir, "fig4b_correlation_matrix.pdf"), panel_final,
       width = 45, height = 23, bg = "white")

message("Figures written to ", fig_dir)
