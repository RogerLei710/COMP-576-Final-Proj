# COMP-576-Final-Proj

This repo is our final project's repository for COMP-576. Please check our report in __Report.pdf__.

In this repo, we devised two __black box attacks__ variants by introducing two novel data augmentation techniques in generating the training data of the substitute model -- Jacobian-based Dataset Augmentation and traditional image Dataset Augmentation. Jacobian-based Dataset Augmentation has more intuitive geometric meaning. Traditional image Dataset Augmentation requires much fewer queries to the target model.

# Remark

Jacobian-based Dataset Augmentation is denoted as __jacobian-beta__ in our code. While the original Jacobian method in [this paper we followed](https://arxiv.org/abs/1602.02697) is denoted as __jacobian-alpha__.
