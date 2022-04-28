基于fairseq实现的翻译模型（不是半自回归的），除了nll loss外，还添加了两个loss：

- 预测每一个词是不是N-gram的开始或者结束的损失；
- Dynamic CRF损失（参考论文《Fast Structured Decoding for Sequence Models》）；

但这两个loss相比较于原本的翻译模型并没有什么帮助，而且现在infer时还有点BUG；

