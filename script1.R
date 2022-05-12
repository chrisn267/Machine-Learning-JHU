library(caret)
library(tidyverse)
library(forecast)
library(lubridate)
library(nnet)

# library(AppliedPredictiveModeling)
# library(pgmm)
# library(elasticnet)
# library(gbm)
# library(rattle)

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml_training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml_testing.csv")

pml_train_full <- read.csv("pml_training.csv")
pml_test_full <- read.csv("pml_testing.csv")

# first 5 rows are information
pml_train <- pml_train_full[,-c(2:5)]
pml_test <- pml_test_full[,-c(2:5)]

# We look at the data and notice a lot of columns are NA
train_na <- data.frame(col1 = names(pml_train), na_col = colSums(is.na(pml_train) | (pml_train == "")))
train_na <- train_na %>% mutate(perc_na = na_col / nrow(pml_train))
train_na

# We realise this is because some rows give summary statistics for each window
pml_train_nw <- pml_train[(pml_train$new_window == "yes"),]
train_na2 <- data.frame(col1 = names(pml_train), na_col = colSums(is.na(pml_train_nw) | (pml_train_nw == "")))
train_na2 <- train_na2 %>% mutate(perc_na = na_col / nrow(pml_train))
train_na2

# We check the test data and realise this data does not include the summary statistics
test_na <- data.frame(col1 = names(pml_test), na_col = colSums(is.na(pml_test) | (pml_test == "")))
test_na <- test_na %>% mutate(perc_na = na_col / nrow(pml_test))
test_na

# So we remove all data with summary statistics
pml_train_slim <- pml_train[,(train_na$na_col == 0)]
pml_test_slim <- pml_test[,(test_na$na_col == 0)]

# Now we ensure classe is a factor
pml_train_slim$classe <- factor(pml_train_slim$classe) 
pml_train_slim$classe <- relevel(pml_train_slim$classe, ref = 5) 

# so we can visualize all data as violin plots
pml_train_slim_narrow <- pivot_longer(pml_train_slim, cols = 4:55, names_to = "var", values_to = "vals")

# g <- ggplot(data = pml_train_slim_narrow, aes(x = vals, color = classe))
# g <- g + geom_density()
# g <- g + facet_wrap(vars(var), scales = "free")
# g <- g + theme(axis.text.x = element_blank(),axis.text.y = element_blank(),
#               panel.background = element_blank())
# g

g <- ggplot(data = pml_train_slim_narrow, aes(y = vals, x = classe, color = classe))
g <- g + geom_violin()
g <- g + facet_wrap(vars(var), scales = "free")
g

g <- g + theme(axis.text.x = element_blank(),axis.text.y = element_blank(),
               panel.background = element_blank())
g

# we check spurious points and remove

checkvar = 'gyros_dumbbell_z' #remove X = 5373
checkvar = 'gyros_belt_x'
checkvar = 'gyros_belt_y'
checkvar = 'gyros_belt_z'
checkvar = 'gyros_forearm_z' #remove num_window = 186 & 187
checkvar = 'magnet_belt_x' #remove num_window = 1,9,10, 404, 863
checkvar = 'magnet_belt_y'
checkvar = 'magnet_belt_z' # remove 2 & (1)
checkvar = 'magnet_dumbbell_y' #remove X = 9274

g <- ggplot(data = pml_train_slim, aes_string(y = checkvar, x = 'num_window', color = 'classe'))
g <- g + geom_point()
g

# var gyros_dumbbell_z (max) remove X = 5373
pml_train_slim$X[which.max(pml_train_slim$gyros_dumbbell_z)]

pml_train_slim <- pml_train_slim[(pml_train_slim$X != '5373'),]
pml_train_slim_narrow <- pivot_longer(pml_train_slim, cols = 4:55, names_to = "var", values_to = "vals")

# var gyros_forearm_z (max and min) remove num_window = 186 & 187
pml_train_slim[order(pml_train_slim$gyros_forearm_z),][1:10,'num_window']
pml_train_slim[order(-pml_train_slim$gyros_forearm_z),][1:10,'num_window']

pml_train_slim <- pml_train_slim[(pml_train_slim$num_window != '186' & pml_train_slim$num_window != '187'),]
pml_train_slim_narrow <- pivot_longer(pml_train_slim, cols = 4:55, names_to = "var", values_to = "vals")

# var magnet_belt_x (max) remove num_window = 1,9,10,404,863
# var magnet_belt_z (max) remove num_window = 2
pml_train_slim_E <- pml_train_slim %>% filter(classe == 'E')
pml_train_slim_E[order(-pml_train_slim_E$magnet_belt_x),][1:50,'num_window']
pml_train_slim_E[order(-pml_train_slim_E$magnet_belt_z),][1:10,'num_window']

pml_train_slim <- pml_train_slim[!(pml_train_slim$num_window %in% c('1','9','10','404','863')),]
pml_train_slim <- pml_train_slim[(pml_train_slim$num_window != '2'),]
pml_train_slim_narrow <- pivot_longer(pml_train_slim, cols = 4:55, names_to = "var", values_to = "vals")

# var magnet_dumbell_y (min) remove X = 9274
pml_train_slim$X[which.min(pml_train_slim$magnet_dumbbell_y)]

pml_train_slim <- pml_train_slim[(pml_train_slim$X != '9274'),]
pml_train_slim_narrow <- pivot_longer(pml_train_slim, cols = 4:55, names_to = "var", values_to = "vals")

# now we have a clean data set so we split into training and test set
# write the following with functions to make it easier to loop through

get_train_data <- function(seed = 1, split = 0.8) {
    set.seed(seed)
    trainIndex <- createDataPartition(pml_train_slim$classe, p = split, list = F)
    data_train <<- pml_train_slim[ trainIndex,-c(1:3)]
    data_val  <<- pml_train_slim[-trainIndex,-c(1:3)]
    data_all <<- pml_train_slim[,-c(1:3)]
    data_test <<- pml_test_slim[,-c(1:3)]
    data_test2 <<- pml_test_slim[,-c(1:3)]
}

set_preProcess <- function(pp = "none") {
    if (pp != "none"){
        model_pp <<- preProcess(data_train, method = pp)
        model_pp2 <<- preProcess(data_all, method = pp)
        data_train <<- predict(model_pp, data_train)
        data_val <<- predict(model_pp, data_val)
        data_all <<- predict(model_pp2, data_all)
        data_test <<- predict(model_pp, data_test)
        data_test2 <<- predict(model_pp2, data_test2)
    }
}

set_trainControl <- function(x) {
    trCont <<- trainControl(method = "repeatedcv", number = 10, repeats = x)
}

fit_model <- function(x) {
    model_fit <<- train(classe ~ ., method = x, trControl = trCont, data = data_train)
    pred_val <<- predict(model_fit, data_val)
    pred_test <<- predict(model_fit, data_test)
    conf_matrix <<- confusionMatrix(pred_val, data_val$classe)
}

fit_full_model <- function(x) {
    model_fit2 <<- train(classe ~ ., method = x, trControl = trCont, data = data_all)
    pred_test2 <<- predict(model_fit2, data_test2)
}

run_routine <- function(seed = 1, split = 0.8, preproc = "center,scale", rpt = 1, method = "knn") {
    counter <- 1
    
    output1 <<- list()
    data_train <<- NULL
    data_val <<- NULL
    data_test <<- NULL
    data_test2 <<- NULL
    data_all <<- NULL
    model_pp <<- NULL
    model_pp2 <<- NULL
    model_fit <<- NULL
    model_fit2 <<- NULL
    pred_val <<- NULL
    pred_test <<- NULL
    pred_test2 <<- NULL
    conf_matrix <<- NULL
    trCont <<- NULL
    
    for (p in preproc){
        for (s in seed){
            for (r in rpt){
                for (l in split){
                    for (m in method){
                        
                        output1[[paste0("data",counter)]] <- list()
                        output1[[paste0("run",counter)]] <- list()
                        
                        output1[[paste0("run",counter)]][["seed"]] <- s
                        output1[[paste0("run",counter)]][["preproc"]] <- p
                        output1[[paste0("run",counter)]][["rpt"]] <- r
                        output1[[paste0("run",counter)]][["split"]] <- l
                        output1[[paste0("run",counter)]][["method"]] <- m
                        
                        starttime <- Sys.time()                        
                        get_train_data(s, l)
                        set_preProcess(p)
                        set_trainControl(r)
                        fit_model(m)
                        fit_full_model(m)
                        endtime <- Sys.time()
                        output1[[paste0("run",counter)]][["duration"]] <- as.numeric(endtime - starttime)
                        
                        output1[[paste0("data",counter)]][["data_train"]] <- data_train
                        output1[[paste0("data",counter)]][["data_val"]] <- data_val
                        output1[[paste0("data",counter)]][["data_test"]] <- data_test
                        output1[[paste0("data",counter)]][["data_test2"]] <- data_test2
                        output1[[paste0("data",counter)]][["data_all"]] <- data_all
                        
                        output1[[paste0("run",counter)]][["model_pp"]] <- model_pp
                        output1[[paste0("run",counter)]][["model_fit"]] <- model_fit
                        output1[[paste0("run",counter)]][["pred_val"]] <- pred_val
                        output1[[paste0("run",counter)]][["pred_test"]] <- pred_test
                        output1[[paste0("run",counter)]][["conf_matrix"]] <- conf_matrix
                        output1[[paste0("run",counter)]][["model_pp2"]] <- model_pp2
                        output1[[paste0("run",counter)]][["model_fit2"]] <- model_fit2
                        output1[[paste0("run",counter)]][["pred_test2"]] <- pred_test2
                        
                        
                        counter <- counter + 1
                    }
                }
            }
        }
    }
    return(output1)
}


reduce1 <- function(x){
    
    outdf <- data.frame(seed = rep("",len),
                        preproc = rep("",len),
                        trCtrl = rep("",len),
                        method = rep("",len),
                        s_accuracy = rep("",len),
                        v_accuracy = rep("",len),
                        duration = rep("",len),
                        m1 = rep("",len),
                        m2 = rep("",len),
                        m3 = rep("",len),
                        m4 = rep("",len),
                        m5 = rep("",len),
                        m6 = rep("",len),
                        m7 = rep("",len),
                        m8 = rep("",len),
                        m9 = rep("",len),
                        m10 = rep("",len),
                        m11 = rep("",len),
                        m12 = rep("",len),
                        m13 = rep("",len),
                        m14 = rep("",len),
                        m15 = rep("",len),
                        m16 = rep("",len),
                        m17 = rep("",len),
                        m18 = rep("",len),
                        m19 = rep("",len),
                        m20 = rep("",len))
    
    for (ii in 1:len) {
        
        rlookup <- paste0("run",ii)
        outdf[ii,"id"] <- ii
        outdf[ii,"seed"] <- x[[rlookup]][["seed"]]
        outdf[ii,"split"] <- x[[rlookup]][["split"]]
        outdf[ii,"preproc"] <- paste(x[[rlookup]][["preproc"]], collapse = ",")
        outdf[ii,"trCtrl"] <- x[[rlookup]][["trCtrl"]]
        outdf[ii,"method"] <- x[[rlookup]][["method"]]
        outdf[ii,"duration"] <- x[[rlookup]][["duration"]]
        outdf[ii,"s_accuracy"] <- as.numeric(mean(x[[rlookup]][["model_fit"]][["resample"]][["Accuracy"]]))
        outdf[ii,"v_accuracy"] <- as.numeric(x[[rlookup]][["conf_matrix"]][["overall"]][1])
        
        for (jj in 1:20) {
            
            clookup <- paste0("m", jj)
            outdf[ii,clookup] <- as.character(x[[rlookup]][["pred_test2"]][jj])
            
        }
    }
    return(outdf)
}


output_no_pp_slim <- reduce1(output_no_pp)
output_cands_slim <- reduce1(output_cands)
output_pca_slim <- reduce1(output_pca)

output_all_slim <- rbind(output_no_pp_slim, output_cands_slim, output_pca_slim)
output_all_slim <- cbind(id = seq(1,72),output_all_slim)
output_all_slim[,c("s_accuracy")] <- as.numeric(output_all_slim[,c("s_accuracy")])
output_all_slim[,c("v_accuracy")] <- as.numeric(output_all_slim[,c("v_accuracy")])

method_list = c("multinom", "naive_bayes", "knn", "rf", "svmLinear", "svmRadial")
method_list2 = c("knn", "rf", "svmRadial")
output_all_slim <- output_all_slim %>% filter(method == method_list)

method_accuracy <- output_all_slim %>% 
                    group_by(method) %>% 
                    summarise(sample_accuracy = mean(s_accuracy), valid_accuracy = mean(v_accuracy)) %>%
                    arrange(desc(valid_accuracy)) %>%
                    ungroup()

preprocess_accuracy <- output_all_slim %>% 
    group_by(preproc) %>% 
    summarise(sample_accuracy = mean(s_accuracy), valid_accuracy = mean(v_accuracy)) %>%
    arrange(desc(valid_accuracy)) %>%
    ungroup()

trCtrl_accuracy <- output_all_slim %>% 
    group_by(trCtrl) %>% 
    summarise(sample_accuracy = mean(s_accuracy), valid_accuracy = mean(v_accuracy)) %>%
    arrange(desc(valid_accuracy)) %>%
    ungroup()

seed_accuracy <- output_all_slim %>% 
    group_by(seed) %>% 
    summarise(sample_accuracy = mean(s_accuracy), valid_accuracy = mean(v_accuracy)) %>%
    arrange(desc(valid_accuracy)) %>%
    ungroup()

mp_accuracy <- output_all_slim %>% 
    group_by(method,preproc) %>% 
    summarise(valid_accuracy = mean(v_accuracy)) %>%
    ungroup()

table(mp_accuracy$method, mp_accuracy$preproc)

accuracy1 <- function(x){
    
    output_all_slim %>% 
        group_by_at(x) %>% 
        summarise(sample_accuracy = mean(s_accuracy), valid_accuracy = mean(v_accuracy)) %>%
        arrange(desc(valid_accuracy)) %>%
        ungroup()
}


output_summary_mean <- matrix(0, nrow = 5, ncol = 20)
for (ii in 1:5) {
    for (jj in 1:20) {
    clookup <- paste0("m", jj)
    output_summary_mean[ii,jj] <- mean(output_all_slim[(output_all_slim[,clookup] == LETTERS[ii]),"v_accuracy"])
    }
}
for (ii in 1:5) {
    for (jj in 1:20) {
        if(is.nan(output_summary_mean[ii,jj])){
            output_summary_mean[ii,jj] <- 0
        } else {
            output_summary_mean[ii,jj] <- round(output_summary_mean[ii,jj],2)
        }
    }
}

output_summary_total <- matrix(0, nrow = 5, ncol = 20)
for (ii in 1:5) {
    for (jj in 1:20) {
    clookup <- paste0("m", jj)
    output_summary_total[ii,jj] <- sum(output_all_slim[(output_all_slim[,clookup] == LETTERS[ii]),"v_accuracy"])
    }
}
for (ii in 1:5) {
    for (jj in 1:20) {
        if(is.nan(output_summary_total[ii,jj])){
            output_summary_total[ii,jj] <- 0
        } else {
            output_summary_total[ii,jj] <- round(output_summary_total[ii,jj],1)
        }
    }
} 

output_summary <- function(x = "sum"){
    
m <- matrix(0, nrow = 5, ncol = 20)

    for (ii in 1:5) {
        for (jj in 1:20) {
            clookup <- paste0("m", jj)
            if (x == "sum") {
                m[ii,jj] <- sum(output_all_slim[(output_all_slim[,clookup] == LETTERS[ii]),"v_accuracy"])
            } else {
                m[ii,jj] <- mean(output_all_slim[(output_all_slim[,clookup] == LETTERS[ii]),"v_accuracy"])
            }
        }
    }
    for (ii in 1:5) {
        for (jj in 1:20) {
            if(is.nan(m[ii,jj])){
                m[ii,jj] <- 0
            } else {
                m[ii,jj] <- round(m[ii,jj],1)
            }
        }
    }

}







output_summary_both <- matrix("", nrow = 2, ncol = 20)
for (jj in 1:20) {
    output_summary_both[1,jj] <- LETTERS[1:5][(which.max(output_summary_mean[,jj]))]
    output_summary_both[2,jj] <- LETTERS[1:5][(which.max(output_summary_total[,jj]))]
}

output_summary_mean
output_summary_total
output_summary_both

method_accuracy
preprocess_accuracy
trCtrl_accuracy
seed_accuracy

View(output_all_slim)

# save data as old for review in appendix
pml_train_slim_old <- pml_train_slim
pml_test_slim_old <- pml_test_slim
pml_train_slim_narrow_old <- pml_train_slim_narrow

# remove data points
pml_train_slim <- pml_train_slim[(pml_train_slim$X != '5373'),]
pml_train_slim <- pml_train_slim[(pml_train_slim$num_window != '186'),]
pml_train_slim <- pml_train_slim[(pml_train_slim$num_window != '187'),]
pml_train_slim <- pml_train_slim[!(pml_train_slim$num_window %in% c('1','9','10','404','863')),]
pml_train_slim <- pml_train_slim[(pml_train_slim$num_window != '2'),]
pml_train_slim <- pml_train_slim[(pml_train_slim$X != '9274'),]

pml_train_slim_narrow <- pivot_longer(pml_train_slim, cols = 4:55, names_to = "var", values_to = "vals")

# plot before and after
plot_data_old <- cbind(dataset = rep("old", nrow(pml_train_slim_narrow_old)), pml_train_slim_narrow_old)
plot_data_new <- cbind(dataset = rep("new", nrow(pml_train_slim_narrow)), pml_train_slim_narrow)
plot_data <- rbind(plot_data_old, plot_data_new)
subset_ <- c("gyros_dumbbell_z", "gyros_forearm_z",
             "magnet_belt_x", "magnet_belt_z", "magnet_dumbbell_y")

plot_data_slim <- plot_data %>% filter(var %in% subset_) %>% filter(dataset == "old")

plot_data_gfz <- plot_data %>% filter(var == "gyros_forearm_z") %>% filter(dataset == "old")
plot_data_gfz[,"var"] <- "gyros_forearm_z_(zoom)"
plot_data_gfz <- plot_data_gfz[(plot_data_gfz$X != '5373'),]

plot_data_use <- rbind(plot_data_slim, plot_data_gfz)

plot_data_use[,"outlier"] <- rep("keep", nrow(plot_data_use))
plot_data_use[(plot_data_use$X == '5373'),"outlier"] <- "gyros_dumbbell_z"
plot_data_use[(plot_data_use$num_window == '186'),"outlier"] <- "gyros_forearm_z"
plot_data_use[(plot_data_use$num_window == '187'),"outlier"] <- "gyros_forearm_z"
plot_data_use[(plot_data_use$num_window %in% c('1','9','10','404','863')),"outlier"] <- "magnet_belt_x"
plot_data_use[(plot_data_use$num_window == '2'),"outlier"] <- "magnet_belt_z"
plot_data_use[(plot_data_use$X == '9274'),"outlier"] <- "magnet_dumbbell_y"

g <- ggplot(data = plot_data_use, aes(y = vals, x = num_window, color = outlier))
g <- g + geom_point(size = 0.5)
g <- g + scale_colour_manual(values = c("keep" = "grey",
                                        "gyros_dumbbell_z" = "red",
                                        "gyros_forearm_z" = "blue",
                                        "magnet_belt_x" = "green",
                                        "magnet_belt_z" = "purple",
                                        "magnet_dumbbell_y" = "orange"))
g <- g + facet_wrap(vars(var),nrow = 2, scales = "free")
# g <- g + theme(axis.text.x = element_blank(),axis.text.y = element_blank(),
#               panel.background = element_blank())
g



method_list_all = c("multinom", "naive_bayes", "knn", "rf", "svmLinear", "svmRadial")
pplist = list("none", c("center","scale"), "pca")

output_summary <- function(x = "sum", ml = method_list_all, pp = pplist) {
    
    m <- matrix(0, nrow = 5, ncol = 20)
    output_slim1 <- output_all_slim %>% filter(method %in% ml) %>% filter(preproc %in% pp)
    
    for (ii in 1:5) {
        for (jj in 1:20) {
            clookup <- paste0("m", jj)
            if (x == "mean") {
                os <- mean(output_slim1[(output_slim1[,clookup] == LETTERS[ii]),"v_accuracy"])
                round_ <- 2
            } else {
                os <- sum(output_slim1[(output_slim1[,clookup] == LETTERS[ii]),"v_accuracy"])
                round_ <- 0
            }
            if (is.nan(os)) {
                m[ii,jj] <- 0
            } else {
                m[ii,jj] <- round(os,round_)
            }
        }
    }
    return(m)
}

output_summary("mean", ml = method_list_all, pp = list("center","scale"))
pplist[2]

output1<- run_routine(seeds = 1,
                      split = 0.7, 
                      preproc = "pca",
                      trctrl = 1,
                      methods = "knn")