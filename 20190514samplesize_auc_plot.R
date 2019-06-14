library(ggplot2)
library(reshape2)

Darts_theme = ggplot2::theme(
  legend.text = ggplot2::element_text(size = 13),
  plot.title = ggplot2::element_text(size=13, face="bold"), 
  axis.title.y = ggplot2::element_text(size=13), 
  axis.title.x = ggplot2::element_text(size=13),
  axis.text.y = ggplot2::element_text(size=13, angle = 90, hjust = 0.5, vjust=0.5), 
  axis.text.x = ggplot2::element_text(size=13, angle=0, hjust=0.5, vjust=0.5),
  legend.background = ggplot2::element_rect(fill = "transparent", colour = "transparent")) +
  ggplot2::theme(
    panel.grid.major = ggplot2::element_blank(), panel.grid.minor = ggplot2::element_blank(),
    panel.background = ggplot2::element_blank(), axis.line = ggplot2::element_line(colour = "black")) +
  ggplot2::theme(panel.border = ggplot2::element_blank(), panel.grid.major = ggplot2::element_blank(),
                 panel.grid.minor = ggplot2::element_blank(), axis.line = ggplot2::element_line(colour = "black"))	


setwd("/Users/connie/Dropbox/deep learning/")
data = read.csv("20190518size_auc_mean_20fold.csv", sep = '\t', header = F)
#std_data = read.csv("20190515size_auc_std.csv", sep = ',', header = F)
cnn = c(.7140000000000001, 0.748, 0.7585, 0.7655, 0.7735000000000001)

data[6,] = cnn

rownames(data) = c("LR","LDA","KNN", "NB","SVM","CNN")
colnames(data) = c(1000, 2000, 3000, 4000, 5000)

#colnames(std_data) = c("LR","LDA","KNN", "NB","SVM","CNN")
#rownames(std_data) = c(1000, 2000, 3000, 4000, 5000, 5930)

df = melt(data)
#df2 = melt(std_data)
df$method = rep(c("LR","LDA","KNN", "NB","SVM","CNN"),5)
#df$std = df2$value

ggplot(data=df, aes(x=variable, y=value, group=method)) +
  geom_line(aes(color=method)) +
  geom_point(aes(color=method)) +
  xlab("Training data set size") + 
  ylab("AUC") +
  Darts_theme
  #geom_errorbar(aes(ymin=value-std, ymax=value+std), width=0.01, size = 0.01) 
