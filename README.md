# Keras Customizable Residual Unit 

This building block shows you how to easily incorporate custom residual connections into your Keras neural networks. I spent longer than I'd have liked trying to add these kind of blocks to my models and thought I'd share it with the world.

Run `python residual.py` to get a model built and a png to peruse.

This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, X., Ren, S., Sun, J., [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027v2). 

Visit the reference implementation at [keras-resnet](https://github.com/raghakot/keras-resnet) to see a full model with bottlenecks and downsampling between units included. 

Further credit to: [Keunwoo Choi](github.com/keunwoochoi), [Nicholas Dronen](github.com/ndronen), and [Alejandro Newell](github.com/anewell) for creating the residual blocks I based this off of.
