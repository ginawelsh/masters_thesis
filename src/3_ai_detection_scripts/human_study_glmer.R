# Experiment 3: frequentist mixed-effects logistic regression with lme4.
# Run analyse_human_study.py first to produce human_study_long.csv, then:
#   Rscript src/3_ai_detection_scripts/human_study_glmer.R
library(lme4)

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d <- read.csv(file.path(here, "human_study_long.csv"))
d$register <- factor(d$register, levels = c("comment", "abstract"))
d$is_human <- as.logical(d$is_human)

m1 <- glmer(correct ~ register + (1 | pid) + (1 | item), data = d, family = binomial)
m2 <- glmer(correct ~ register * is_human + (1 | pid) + (1 | item), data = d, family = binomial)
print(summary(m1))
print(summary(m2))
print(confint(m1, method = "Wald"))

# Likelihood-ratio test of the register effect (m1 against the intercept-only model)
m0 <- glmer(correct ~ 1 + (1 | pid) + (1 | item), data = d, family = binomial)
print(anova(m0, m1))
print(confint(m2, method = "Wald"))
