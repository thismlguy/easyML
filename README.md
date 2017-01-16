#easyML - An python packge to streamline data analysis tasks [built on sklearn]

**easyML** is a python package designed to streamline the process of analyzing datasets using predictive models. It covers crutial aspects the of data analysis process starting from preprocessing, to feature engineering and finally predictive modeling.

They key advantage of this package is the DataBlock module using which you can create a block of your data at the start of analysis. The other modules take this block as input and seemlessly work on your data together. It definitely comes at the cost of loss of generalization as compared to the raw scikit-learn features, but the idea is to incorporate typically used actions and also provide options for flexibility through user-defined tasks. The package is particularly useful for begineers and intermediate level python data science enthusiasts, who are looking to get the job done without worrying about the code.

## Important Links
HTML Documentation - Coming soon!
Examples - Coming soon!

## Installation and Usage
The module can be installed using Github or PyPi as:

## Motivation - How it all began?

I would like to share my motivation behind making this package. I started my data science career using R and SAS but then decided to switch to Python about a year and a half back. I started with the scikit-learn package and I was taken aback by the amount of coding required. In R, when you build a model in caret, it gives you a lot of information required to analyze them. But here, everything has to be coded separately. After some struggle, I decided to create wrapper functions for myself which allowed me to re-use most of the code that I wrote and not worry about looking for syntaxes every time I start a new analysis. My library developed over time.

Then I joined the MS in Data Science program at Columbia and got busy with academics. Returning to Python after a break of ~6 months, my library came in very handy when I had to do a quick project as I had forgotten most of the syntaxes. Doing everything from scratch would have been a nightmare. Its then that I realized that if its so useful for me, I should open-source my work. This is just the beginning, there's lots more to come. 

## How can you contribute?

What you see now is an alpha-version of the package. I have defined a framework with a class structure having clear dataflow. The module contains some fundamental modules. There's still a lot to be done as there are lots of cool modules to be added. I have my upcoming functions list in each module.

I'm sure if you are a serious analyst/data scientist, you would have many such wrappers of your own. I would love to hear about them and it'll be great if you contrinute your ideas/code. Even if you don't have the code, feel free to add feature requests for things you would like see.
