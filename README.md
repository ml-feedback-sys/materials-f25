# Machine Learning in Feedback Systems
*Cornell CS 6784 Fall 2025* 

Meeting Tu/Thu 10:10-11:25am in 366 Hollister Hall\
Professor [Sarah Dean](https://sdean.website/), office hours after lecture by appointment\
TA [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara), office hours by appointment (see Ed)\
Please use [Ed Discussions](https://edstem.org/us/courses/85093/discussion) for any questions.

Please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLScSIJy9JsKhzBEBKLcxumFdUrlA540Uj1ETl4lE4hWHz_fEIQ/viewform?usp=dialog) to join the class's [collaborative GitHub repository](https://github.com/ml-feedback-sys/collaborative-f25).\
If you are on the waitlist, please fill out [this interest form](https://docs.google.com/forms/d/e/1FAIpQLScks25ekpabALpCronYx9ZV_thu6Besk69mRnHcbsuLrh6psA/viewform?usp=sharing&ouid=113616522367506075718).

## Description
Feedback&mdash;when information about system output is used in system input&mdash;plays an important role in machine learning.
For example, models are trained with iterative algorithms which reduce errors on observed data.
Autoregressive models use feedback to capture patterns in sequential data. 
And feedback between machine learning models and the environment in which they are deployed leads to a host of challenges, from distribution shift to bias to polarization. 
This graduate level course will introduce theoretical foundations for studying such phenomena. We will cover the frameworks of online/adaptive learning,  control theory, and reinforcement learning. For each, we will discuss algorithms for ensuring properties like stability, robustness, safety, and fairness. We will also discuss the social and ethical concerns which motivate these algorithms and properties. Student presentations and a research project are major parts of the course.

### Topics and Schedule
Unit 1: Prediction (Aug-Sept)\
*Topics: Supervised Learning, Online Learning, Dynamical Systems, State Estimation. Sequence and timeseries models.*\
Unit 2: Action (Sept-Oct)\
*Topics: Multi-Armed Bandits, Optimal Control & Reinforcement Learning, Model Predictive Control*

The detailed [calendar](calendar.md) will be updated throughout the semester. List of references and papers will be posted in September.

### Prerequisites

Knowledge of ML at the level of CS4780 is recommended. Perhaps more important is mathematical maturity and a working understanding of linear algebra, convex optimization, and probability. The following references may be useful to review: [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf), [Convex Optimization Overview](https://cs229.stanford.edu/section/cs229-cvxopt.pdf), and [Review of Probability Theory](https://cs229.stanford.edu/section/cs229-prob.pdf).

## Assignments
Students will complete weekly assignments, give a presentation, and work on a project during the semester. Depending on enrollment, some of this work may be done in pairs or groups.

### Weekly Assignments 
Assignments will be posted each week on Thursday and are due the following Thursday. We will use GitHub collaboration tools to manage and collect your work.

### Presentations
Throughout the semester, students will present selected papers or textbook excerpts (list to come) and lead a discussion. Students are required to schedule a meeting with the TA to go over their presentation at least two days before they are scheduled to present. [Presentation Details](presentation.md).


### Final Project
Projects can be done in groups of up to four, with expectations scaling with the size of the group. Students are encouraged to propose a topic that connects class material to their research. The deliverables are:
 - Project proposal (1 page) due in October
 - Midterm update (3 pages) due in November
 - Project report (5-6 pages) due at the end of the semester

[Final Project Details](project.md).

### Grading
Students will be evaluated by:
 - 40% final project
 - 25% paper presentation
 - 25% weekly assignments
 - 10% participation
