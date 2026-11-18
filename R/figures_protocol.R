# Figures of the protocol: single split vs nested CV (fig 5a), dual path (5b),
# negative control (6) and a few supplementary panels.
# Needs results/{real,null}/summary/ from the notebook. Run: Rscript R/figures_protocol.R
source(here::here("R", "00_setup.R"))

col_single <- "#D55E00"; col_nested <- "#0072B2"

# csv names -> labels (order = facet order)
model_labels <- c(Cox_RIDGE = "Ridge", Cox_LASSO = "LASSO", RSF = "RSF", GBM = "GBSA", SSVM = "SSVM")

### Figure 5a - single split vs nested CV
d <- read_result("real", "summary", "delta_by_seed.csv")
stopifnot(all(unique(d$model) %in% names(model_labels)))

lev <- c("Single split\n(70/30)", "Nested CV\n(mean of 5 folds)")

# (i) Main panel: Cox Ridge, Path B
fig <- d %>%
  filter(path == "B", model == "Cox_RIDGE", metric == "ci_test") %>%
  select(seed, single, nested_mean) %>%
  drop_na() %>%
  pivot_longer(c(single, nested_mean), names_to = "esquema", values_to = "cindex") %>%
  mutate(esquema = factor(esquema, levels = c("single", "nested_mean"), labels = lev))

res <- fig %>%
  group_by(esquema) %>%
  summarise(m = mean(cindex), s = sd(cindex), .groups = "drop")

p_ridge <- ggplot(fig, aes(esquema, cindex)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_line(aes(group = seed), colour = "grey75", linewidth = 0.4) +
  geom_point(aes(colour = esquema), size = 2.4) +
  geom_pointrange(data = res, aes(x = esquema, y = m, ymin = m - s, ymax = m + s),
                  inherit.aes = FALSE, position = position_nudge(x = 0.22),
                  colour = "black", linewidth = 0.6, size = 0.5) +
  geom_text(data = res, aes(x = esquema, y = min(fig$cindex) - 0.03,
                            label = sprintf("SD = %.3f", s)),
            inherit.aes = FALSE, size = 7) +
  scale_colour_manual(values = c(col_single, col_nested), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.12, 0.05))) +
  labs(x = NULL, y = "C-index (test)") +
  theme_classic(base_size = 24) +
  theme(axis.text  = element_text(size = 24, colour = "black"),
        axis.title = element_text(size = 25))

# save_fig(p_ridge, "fig5a_single_vs_nested_ridge_pathB", width = 3.8, height = 4,
#          formats = c("pdf", "png"))

# (ii) All models and both paths
fig_s <- d %>%
  filter(metric == "ci_test") %>%
  select(seed, path, model, single, nested_mean) %>%
  drop_na() %>%
  pivot_longer(c(single, nested_mean), names_to = "esquema", values_to = "cindex") %>%
  mutate(esquema = factor(esquema, levels = c("single", "nested_mean"),
                          labels = c("Single", "Nested")),
         path  = factor(path, labels = c("Path A", "Path B")),
         model = factor(model, levels = names(model_labels), labels = model_labels))

res_s <- fig_s %>%
  group_by(path, model, esquema) %>%
  summarise(m = mean(cindex), s = sd(cindex), .groups = "drop")

p_all <- ggplot(fig_s, aes(esquema, cindex)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_line(aes(group = seed), colour = "grey75", linewidth = 0.6) +
  geom_point(aes(colour = esquema), size = 2.5) +
  geom_pointrange(data = res_s, aes(x = esquema, y = m, ymin = m - s, ymax = m + s),
                  inherit.aes = FALSE, position = position_nudge(x = 0.25),
                  linewidth = 0.5, size = 0.4) +
  scale_colour_manual(
    values = c(Single = col_single, Nested = col_nested),
    labels = c(Single = "Single split", Nested = "Nested CV"),
    name   = NULL) +
  guides(colour = guide_legend(override.aes = list(size = 3.5))) +
  facet_grid(path ~ model) +
  labs(x = NULL, y = "C-index (test)") +
  theme_bw(base_size = 22) +
  theme(panel.grid.minor = element_blank(),
        axis.text  = element_text(size = 24, colour = "black"),
        axis.title = element_text(size = 25),
        strip.text = element_text(size = 25),
        axis.text.x  = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position   = "top",
        legend.text       = element_text(size = 25),
        legend.background = element_rect(colour = "black", fill = "white", linewidth = 0.6),
        legend.margin     = margin(5, 10, 5, 10))

save_fig(p_all, "fig5a_single_vs_nested_all_models",         width = 18, height = 9)
# save_fig(p_all, "fig5a_single_vs_nested_all_models_compact", width = 9,  height = 5)

### Figure 5b - dual path: Path A - Path B, per seed (paired by outer fold)
# three ticks per panel: -b, 0, b
breaks3 <- function(lims) {
  m <- min(abs(lims))
  e <- 10^floor(log10(m))
  b <- floor(m / e + 1e-9) * e
  c(-b, 0, b)
}

dp <- read_result("real", "summary", "dual_path_by_seed.csv")
stopifnot(all(unique(dp$model) %in% names(model_labels)))

dp <- dp %>%
  mutate(advantage_A = if_else(metric == "ibs", -diff_mean, diff_mean),   # IBS: lower is better
         metric = factor(metric, levels = c("ci_test", "auc", "ibs"),
                         labels = c("C-index", "AUC", "IBS")),
         model  = factor(model, levels = rev(names(model_labels)),
                         labels = rev(model_labels))) %>%
  drop_na(advantage_A)

means <- dp %>%
  group_by(metric, model) %>%
  summarise(m = mean(advantage_A), .groups = "drop")

p_dual <- ggplot(dp, aes(advantage_A, model)) +
  scale_y_discrete(drop = FALSE) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "grey93") +
  geom_vline(xintercept = 0, colour = "grey30", linewidth = 0.8) +
  geom_point(position = position_jitter(height = 0.15, width = 0, seed = 1),
             colour = "grey35", alpha = 0.7, size = 3.5) +
  geom_point(data = means, aes(m, model), shape = 18, size = 8, colour = "black") +
  scale_x_continuous(breaks = breaks3) +
  facet_wrap(~ metric, scales = "free_x") +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 24) +
  theme(panel.grid.minor = element_blank(),
        axis.text  = element_text(size = 20, colour = "black"),
        axis.title = element_text(size = 24),
        strip.text = element_text(size = 25),
        panel.spacing = unit(1.2, "lines"))

save_fig(p_dual, "fig5b_dual_path", width = 8, height = 8, format = "pdf")

### Figure 6 - negative control: radiomic features vs Gaussian noise
real_s <- read_result("real", "summary", "delta_by_seed.csv")
nul_s  <- read_result("null", "summary", "delta_by_seed.csv")

# TRUE : Path A (real and noise) over the folds with a non-empty selection -> as the margin of Table 3
# FALSE: Path A real over all folds ("Nested CV" column of Table 3); noise stays non-empty
real_A_no_vacios <- TRUE

col_dat  <- c("Radiomic features" = "#009E73", "Gaussian noise features" = "#CC79A7")
lab_path <- c(A = "Path A", B = "Path B")
met_lab  <- c(ci_test = "C-index", auc = "AUC", ibs = "IBS")

tema <- theme_bw(base_size = 16) +
  theme(panel.grid.minor = element_blank(),
        axis.text    = element_text(size = 19, colour = "black"),
        axis.title   = element_text(size = 19),
        strip.text   = element_text(size = 18),
        legend.text  = element_text(size = 18),
        legend.title = element_blank(),
        legend.position = "top",
        plot.title   = element_text(size = 19),
        plot.tag     = element_text(size = 22, face = "bold"))

# value per seed: path A over non-empty folds (empty ones count as 0.5), path B over all folds
seed_values <- function(d, nonempty_A = TRUE) {
  k_seed <- d %>%
    filter(path == "A", metric == "ibs", model != "SSVM") %>%    # IBS n_folds = folds with a fitted model
    group_by(seed) %>%
    summarise(kmin = min(n_folds), k = max(n_folds), .groups = "drop")
  stopifnot(all(k_seed$kmin == k_seed$k))

  A <- d %>% filter(path == "A") %>%
    left_join(select(k_seed, seed, k), by = "seed") %>%
    mutate(n_empty  = 5 - k,
           n_def_ne = if_else(metric == "ibs", n_folds, n_folds - n_empty),
           value = case_when(
             !nonempty_A & metric != "ibs" ~ nested_mean,           # all folds
             k == 0 | n_def_ne <= 0        ~ NA_real_,
             metric == "ibs"               ~ nested_mean,
             TRUE                          ~ (n_folds * nested_mean - n_empty * 0.5) / n_def_ne))

  B <- d %>% filter(path == "B") %>% mutate(value = nested_mean)

  bind_rows(A, B) %>% select(seed, path, model, metric, value)
}

resumir <- function(d, etiqueta, nonempty_A) {
  seed_values(d, nonempty_A) %>%
    filter(!is.na(value)) %>%
    group_by(path, model, metric) %>%
    summarise(n_seeds = n(), nested_mean = mean(value), sd = sd(value), .groups = "drop") %>%
    mutate(datos = etiqueta)
}

res_nc <- bind_rows(resumir(real_s, "Radiomic features",       real_A_no_vacios),
                    resumir(nul_s,  "Gaussian noise features", TRUE)) %>%
  mutate(model = case_when(model == "Cox_RIDGE" ~ "Cox Ridge",
                           model == "Cox_LASSO" ~ "Cox LASSO",
                           model == "GBM"       ~ "GBSA",
                           TRUE                 ~ model),
         model  = factor(model, levels = c("Cox Ridge", "Cox LASSO", "RSF", "GBSA", "SSVM")),
         metric = factor(metric, levels = names(met_lab), labels = met_lab))

# References: 0.5 for C-index / AUC (chance), 0.25 for the IBS (non-informative model)
ref <- tibble(metric = factor(met_lab, levels = met_lab), y = c(0.5, 0.5, 0.25))

p_neg <- ggplot(res_nc, aes(model, nested_mean, colour = datos)) +
  geom_hline(data = ref, aes(yintercept = y), linetype = "dashed", colour = "grey50") +
  geom_pointrange(aes(ymin = nested_mean - sd, ymax = nested_mean + sd),
                  position = position_dodge(width = 0.55), size = 0.7, linewidth = 0.8,
                  na.rm = TRUE) +
  scale_colour_manual(values = col_dat) +
  facet_grid(metric ~ path, scales = "free_y",
             labeller = labeller(path = lab_path)) +
  labs(x = NULL, y = "Nested CV") +
  tema +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "bottom")

save_fig(p_neg, "fig6_negative_control", width = 8, height = 7.5)

### Table 3 helper - Path A restricted to the non-empty folds, under noise
k_seed_null <- nul_s %>%
  filter(path == "A", metric == "ibs", model != "SSVM") %>%
  group_by(seed) %>%
  summarise(k = max(n_folds), .groups = "drop")

pathA_sel <- nul_s %>%
  filter(path == "A") %>%
  left_join(k_seed_null, by = "seed") %>%
  filter(k > 0) %>%
  mutate(n_empty  = 5 - k,
         n_def_ne = if_else(metric == "ibs", n_folds, n_folds - n_empty),
         value    = case_when(
           n_def_ne <= 0   ~ NA_real_,
           metric == "ibs" ~ nested_mean,
           TRUE            ~ (n_folds * nested_mean - n_empty * 0.5) / n_def_ne)) %>%
  filter(!is.na(value)) %>%
  mutate(model = case_when(model == "Cox_RIDGE" ~ "Cox Ridge",
                           model == "Cox_LASSO" ~ "Cox LASSO",
                           model == "GBM"       ~ "GBSA",
                           TRUE                 ~ model))

# Summary per model and metric (mean weighted by the number of non-empty folds)
table3_noise_A <- pathA_sel %>%
  group_by(metric, model) %>%
  summarise(n_seeds = n(), n_folds_sel = sum(k),
            mean = weighted.mean(value, k),
            min = min(value), max = max(value), .groups = "drop")
print(table3_noise_A, n = Inf)
write_delim(table3_noise_A, file.path(results_dir, "tables", "table3_pathA_nonempty_noise.csv"), delim = ";")

message("Figures written to ", fig_dir)
