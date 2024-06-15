install.packages("PtProcess")
library("PtProcess")
data <- read.csv("usgs_sumatra_fix_15.csv") #time(2004-2023) lat(-6,6) long(94.5,109.5) mag(>=4) depth(<=60) N=7500 T=7305
start_time <- as.POSIXct("2004-01-01T00:00:00.000Z", format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")
# Convert the 'time' column to POSIXct
data$time_posix <- as.POSIXct(data$time, format="%Y-%m-%dT%H:%M:%OSZ", tz="UTC")
# Apply the continuous value calculation
data$continuous_value <- sapply(data$time_posix, function(x) {
  as.numeric(difftime(x, start_time, units = "secs")) / 86400
})
filtered_data <- data[c("time", "latitude", "longitude", "depth", "mag", "continuous_value")]
filtered_data$mag <- filtered_data$mag - 3.95 # mag(>=4)
names(filtered_data)[names(filtered_data) == "time"] <- "timestamp"
names(filtered_data)[names(filtered_data) == "continuous_value"] <- "time"
filtered_data = filtered_data[filtered_data$mag >0, ]
names(filtered_data)[names(filtered_data) == "mag"] <- "magnitude"
dmagn_mark <- function(x, data, params){
  #  Gamma distribution
  #  exponential density when params[7]=0
  if (params[7]>0){
    lambda <- etas_gif(data, x[,"time"], params=params[1:5])
    y <- dgamma(x[,"magnitude"], shape=1+sqrt(lambda)*params[7],
                rate=params[6], log=TRUE)
  } else y <- dexp(x[,"magnitude"], rate=params[6], log=TRUE)
  return(y)
}
rmagn_mark <- function(ti, data, params){
  #  Gamma distribution
  #  exponential density when params[7]=0
  if (params[7]>0){
    lambda <- etas_gif(data, ti, params=params[1:5])
    y <- rgamma(1, shape=1+sqrt(lambda)*params[7],
                rate=params[6])
  } else y <- rexp(1, rate=params[6])
  return(list(magnitude=y))
}
TT <- c(0, 7305)
params <- c(0.001, 0.01, 1, 0.01, 1.3, 1/mean(filtered_data$magnitude), 0)
x <- mpp(data=filtered_data, gif=etas_gif,
         mark=list(dmagn_mark, rmagn_mark),
         params=params, TT=TT,
         gmap=expression(params[1:5]),
         mmap=expression(params))
expmap <- function(y, p){
  #   for exponential distribution
  y$params[1:5] <- exp(p)
  return(y)
}
initial <- log(params[1:5])
z <- optim(initial, neglogLik, object=x, pmap=expmap,
           control=list(trace=1, maxit=100))
initial <- z$par
z <- nlm(neglogLik, initial, object=x, pmap=expmap,
         print.level=2, iterlim=500, typsize=initial)
x0 <- expmap(x, z$estimate)
print(logLik(x0))
allmap <- function(y, p){
  y$params <- exp(p)
  return(y)
}
initial <- log(c(0.001, 0.01, 1, 0.01, 1.3, 1/mean(filtered_data$magnitude), 0.1))
z <- optim(initial, neglogLik, object=x, pmap=allmap,
           control=list(trace=1, maxit=200))
initial <- z$par
z <- nlm(neglogLik, initial, object=x, pmap=allmap,
         print.level=2, iterlim=500, typsize=initial)
x1 <- allmap(x, z$estimate)
print(logLik(x1))
print(summary(x0))
print(summary(x1))
param.est <- rbind(cbind(x0$params, x1$params), c(logLik(x0), logLik(x1)))
colnames(param.est) <- c("Exp Model", "Gamma Model")
rownames(param.est) <- c("p1=mu", "p2=A", "p3=alpha", "p4=c", "p5=p", "p6", "p7", "logLik")
print(param.est)
x1$data$grid_label <- paste(
  "Grid",
  floor(x1$data$latitude),
  "_",
  floor(x1$data$longitude),
  sep = ""
)
x1$data$grid_label <- paste(
  "Grid",
  floor(x1$data$latitude),
  "_",
  ifelse(
    (x1$data$longitude - floor(x1$data$longitude)) >= 0.5,
    ceiling(x1$data$longitude),
    floor(x1$data$longitude)
  ),
  sep = ""
)
grid_labels <- vector("character")  # Initialize an empty character vector to store grid labels
for (x in -6:5) {
  for (y in 95:109) {
    grid_label <- paste("Grid", x, "_", y, sep = "")
    grid_labels <- c(grid_labels, grid_label)  # Append the new grid label to the list
  }
}
all_outputs <- list()
TT_values <- seq(0, 7304, by = 1)
for(TT_val in TT_values) {
  # Initialize a list for each TT value
  all_outputs[[paste("TT", TT_val+1, sep = "_")]] <- list()
  for(grid in grid_labels) {
    # Subset data for current grid
    current_data <- subset(x1$data, grid_label == grid)
    # Apply etas_gif function and store the result
    result <- etas_gif(data = current_data, params = x1$params[1:5], TT = c(TT_val, TT_val + 1))
    # Store the result in the corresponding TT list
    all_outputs[[paste("TT", TT_val+1, sep = "_")]][[grid]] <- result
  }
}
max_values_per_TT <- vector("numeric", length = length(names(all_outputs)))
names(max_values_per_TT) <- names(all_outputs)  # Naming the vector elements
for(TT_val in names(all_outputs)) {
  max_value <- -Inf  # Start with the lowest possible number
  # Iterate through each grid to find the max value
  for(grid in names(all_outputs[[TT_val]])) {
    current_value <- all_outputs[[TT_val]][[grid]]
    if (!is.na(current_value) && current_value > max_value) {
      max_value <- current_value
    }
  }
  # Store the max value for this TT iteration
  max_values_per_TT[TT_val] <- max_value
}
max_values_df <- data.frame(TT = names(max_values_per_TT), MaxValue = max_values_per_TT)
max_values_df$log_MaxValue <- log(max_values_df$MaxValue)
max_values_df$TT_numeric <- as.numeric(gsub("TT_", "", max_values_df$TT))
write.csv(max_values_df, "max_values_per_TT_sumatra_fix_15.csv", row.names = FALSE)
# Initialize an empty data frame
grid_columns <- c()
for (A in -6:5) {
  for (B in 95:109) {
    grid_columns <- c(grid_columns, paste("Grid", A, "_", B, sep = ""))
  }
}
data_to_save <- data.frame(matrix(ncol = length(grid_columns), nrow = 7305))
names(data_to_save) <- grid_columns
# Populate the data frame
for (TT in 1:7305) {
  TT_label <- paste("TT", TT, sep = "_")
  for (A in -6:5) {
    for (B in 95:109) {
      grid_label <- paste("Grid", A, "_", B, sep = "")
      data_to_save[TT, grid_label] <- all_outputs[[TT_label]][[grid_label]]
    }
  }
}
write.csv(data_to_save, "matrix_data_CNN_sumatra_fix_15.csv", row.names = FALSE)