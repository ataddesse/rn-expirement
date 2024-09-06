import { Asset } from "expo-asset";
import { InferenceSession, TypedTensor } from "onnxruntime-react-native";
import { createModelInput } from "./tokenizer/BertProcessor";
import { BertTokenizer, CLS_INDEX, SEP_INDEX, loadTokenizer } from "./tokenizer/BertTokenizer";

export class TransformerOpsInitException extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TransformerOpsInitException";
  }
}

export class TransformerOps {
  public onnxModel?: InferenceSession;
  public tokenizer?: BertTokenizer;
  private static instance?: TransformerOps;
  private static initPromise?: Promise<TransformerOps>;

  public static initialized = TransformerOps.initPromise !== undefined;

  public constructor(model: InferenceSession, tokenizer: BertTokenizer) {
    this.onnxModel = model;
    this.tokenizer = tokenizer;
  }

  public static getInstance(): Promise<TransformerOps> {
    if (!TransformerOps.instance) {
      if (!TransformerOps.initPromise) {
        TransformerOps.initPromise = TransformerOps.init();
      }
      return TransformerOps.initPromise!;
    }
    return Promise.resolve(TransformerOps.instance);
  }

  public static async init(): Promise<TransformerOps> {
    if (!TransformerOps.initPromise) {
      const onnxModel = await TransformerOps.loadOnnxModel();
      const tokenizer = await loadTokenizer();

      if (!onnxModel || !tokenizer) {
        throw new TransformerOpsInitException(
          "TransformerOps: Initialization failed - could not load onnxModel or tokenizer"
        );
      }

      const transformerOps = new TransformerOps(onnxModel, tokenizer);
      TransformerOps.instance = transformerOps;
      return transformerOps;
    }
    return TransformerOps.initPromise!;
  }

  public static async loadOnnxModel(): Promise<InferenceSession> {
    const assets = await Asset.loadAsync(require("./assets/distil-quad/model.onnx"));
    const modelUri = assets[0].localUri;
    if (!modelUri) {
      throw new TransformerOpsInitException(
        "TransformerOps: Initialization failed - could not find model.onnx"
      );
    }
    const model = await InferenceSession.create(modelUri);
    return model;
  }

  /**
   * Utility function to apply softmax to logits.
   * 
   * @param logits - The logits array.
   * @returns - Softmax probabilities.
   */
  softmax(logits: Float32Array): Float32Array {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    return exps.map(exp => exp / sumExps);
  }

  /**
   * Mask undesired tokens by setting their logits to a large negative value.
   * 
   * @param logits - The logits array.
   * @param undesiredTokens - A binary array indicating undesired token positions.
   * @returns - Logits array with undesired tokens masked.
   */
  maskUndesiredTokens(logits: Float32Array, undesiredTokens: Uint8Array): Float32Array {
    return logits.map((logit, idx) => undesiredTokens[idx] === 1 ? -10000.0 : logit);
  }

  /**
   * Decode valid spans based on start and end logits, ensuring span constraints.
   * 
   * @param startLogits - Softmaxed start logits.
   * @param endLogits - Softmaxed end logits.
   * @param topk - Number of top spans to consider.
   * @param maxAnswerLength - The maximum answer length.
   * @param undesiredTokens - Mask to filter out invalid tokens.
   * @returns - The best start, end positions and their corresponding score.
   */
  decodeSpans(
    startLogits: Float32Array, 
    endLogits: Float32Array, 
    topk: number, 
    maxAnswerLength: number, 
    undesiredTokens: Uint8Array
  ): { starts: number[], ends: number[], scores: Float32Array } | null {

    const outer = new Float32Array(startLogits.length * endLogits.length);

    // Compute the score of each tuple(start, end) as outer product
    for (let i = 0; i < startLogits.length; i++) {
      for (let j = 0; j < endLogits.length; j++) {
        outer[i * endLogits.length + j] = startLogits[i] * endLogits[j];
      }
    }

    // Remove invalid candidates where end < start or end - start > maxAnswerLength
    const candidates = outer.map((val, idx) => {
      const startIdx = Math.floor(idx / endLogits.length);
      const endIdx = idx % endLogits.length;
      return (endIdx >= startIdx && endIdx - startIdx < maxAnswerLength && undesiredTokens[startIdx] === 0 && undesiredTokens[endIdx] === 0) ? val : -Infinity;
    });

    // Find top-k spans by sorting candidates
    const scoresFlat = Array.from(candidates);
    let idxSort: number[];
    if (topk === 1) {
      idxSort = [scoresFlat.indexOf(Math.max(...scoresFlat))];
    } else {
      idxSort = Array.from(scoresFlat.keys()).sort((a, b) => scoresFlat[b] - scoresFlat[a]).slice(0, topk);
    }

    const starts = idxSort.map(idx => Math.floor(idx / endLogits.length));
    const ends = idxSort.map(idx => idx % endLogits.length);

    // Filter out invalid spans (should be unnecessary due to earlier filtering)
    const validSpans = starts.filter((_, idx) => candidates[starts[idx] * endLogits.length + ends[idx]] > -Infinity);
    const filteredStarts = validSpans.map(idx => starts[idx]);
    const filteredEnds = validSpans.map(idx => ends[idx]);

    const scores = new Float32Array(filteredStarts.map((start, idx) => candidates[start * endLogits.length + filteredEnds[idx]]));

    return { starts: filteredStarts, ends: filteredEnds, scores };
  }

  public async runInference(question: string): Promise<string | null> {
    const context = `Austin is the Capital of Texas`;
    try {
      // Tokenize the question and context
      const tokenizedQuestion = this.tokenizer!.tokenize(question);
      const tokenizedContext = this.tokenizer!.tokenize(context);
  
      // Log tokenized input
      console.log("Tokenized Question: ", tokenizedQuestion);
      console.log("Tokenized Context: ", tokenizedContext);
  
      // Create input sequence with [CLS], [SEP], etc.
      const inputTokens = [
        CLS_INDEX, 
        ...tokenizedQuestion, 
        SEP_INDEX, 
        ...tokenizedContext, 
        SEP_INDEX
      ];
  
      // Create ONNX input tensors
      const onnxInput = await createModelInput(inputTokens);
  
      // Run the model and get start and end logits
      const modelResponse: InferenceSession.ReturnType = await this.onnxModel!.run(onnxInput);
  
      // Log model response
      console.log("Model Response: ", modelResponse);
  
      const startLogitsTensor = modelResponse["start_logits"] as TypedTensor<"float32">;
      const endLogitsTensor = modelResponse["end_logits"] as TypedTensor<"float32">;
  
      // Log logits before softmax
      console.log("Start Logits (raw): ", startLogitsTensor.data);
      console.log("End Logits (raw): ", endLogitsTensor.data);
  
      const startLogits = startLogitsTensor.data;
      const endLogits = endLogitsTensor.data;
  
      // Create undesired tokens mask (e.g., CLS, SEP, and padding tokens)
      const undesiredTokens = new Uint8Array(inputTokens.length);
      undesiredTokens.fill(0);
      undesiredTokens[0] = 1; // Mark CLS as undesired
      undesiredTokens[tokenizedQuestion.length + 1] = 1; // Mark first SEP as undesired
      undesiredTokens[tokenizedQuestion.length + tokenizedContext.length + 2] = 1; // Mark second SEP as undesired
  
      // Mask undesired tokens by setting their logits to a large negative value before softmax
      const maskedStartLogits = this.maskUndesiredTokens(startLogits, undesiredTokens);
      const maskedEndLogits = this.maskUndesiredTokens(endLogits, undesiredTokens);
  
      // Softmax normalization after masking undesired tokens
      const startProbs = this.softmax(maskedStartLogits);
      const endProbs = this.softmax(maskedEndLogits);
  
      // Log softmax results
      console.log("Softmax: startProbs: ", startProbs);
      console.log("Softmax: endProbs: ", endProbs);
  
      // Decode the best spans with filtering based on valid constraints
      const bestSpan = this.decodeSpans(startProbs, endProbs, 1, 30, undesiredTokens);
  
      if (!bestSpan || bestSpan.starts.length === 0 || bestSpan.ends.length === 0) {
        console.error("No valid answer found");
        return null;
      }
  
      // Log the best span
      console.log("Best Span: ", bestSpan.starts[0], bestSpan.ends[0]);
  
      // Extract the tokens within the best span
      const answerTokens = inputTokens.slice(bestSpan.starts[0], bestSpan.ends[0] + 1);
  
      console.log("Answer Tokens: ", answerTokens);
  
      const answer = this.tokenizer!.decode(answerTokens.filter(token => token !== SEP_INDEX && token !== CLS_INDEX));
  
      console.log("Final Answer: ", answer);
      
      return answer;
    } catch (error) {
      console.error("TransformerOps: Failed to process question", error);
      return null;
    }
  }  
}