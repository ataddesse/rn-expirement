# Tokenizer Directory

This directory contains `BertTokenizer.ts` and `BertProcessor.ts`, both adapted for React Native.
The tokenizer ([original source](https://raw.githubusercontent.com/tensorflow/tfjs-models/master/qna/src/bert_tokenizer.ts))
and processor ([original source](https://github.com/cassiebreviu/bert-excel-addin-ort-web/blob/e470996f33f27edd07283326f934e96f25fab3b6/src/functions/bert/bertProcessing.ts)) have been modified to remove dependencies on TensorFlow.js and for use in react native environment.
