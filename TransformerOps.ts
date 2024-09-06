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
   * Decode valid spans based on start and end logits, while ensuring constraints.
   * 
   * @param startLogits - Softmaxed start logits.
   * @param endLogits - Softmaxed end logits.
   * @param maxAnswerLength - The maximum answer length.
   * @param undesiredTokens - Mask to filter out invalid tokens (e.g., padding or [CLS]).
   * @returns - The best start, end positions and corresponding score.
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
      return (endIdx >= startIdx && endIdx - startIdx < maxAnswerLength) ? val : -Infinity;
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

    // Filter out undesired tokens (e.g., [CLS], [SEP])
    const desiredSpans = starts.map((start, idx) => undesiredTokens[start] === 0 && undesiredTokens[ends[idx]] === 0);
    const filteredStarts = starts.filter((_, idx) => desiredSpans[idx]);
    const filteredEnds = ends.filter((_, idx) => desiredSpans[idx]);

    const scores = new Float32Array(filteredStarts.map((start, idx) => candidates[start * endLogits.length + filteredEnds[idx]]));

    return { starts: filteredStarts, ends: filteredEnds, scores };
  }

  /**
   * Main function to run inference on a question and context.
   * 
   * @param question - The question string.
   * @param context - The context string where the answer is searched.
   * @returns - The extracted answer or null if not found.
   */
  public async runInference(question: string): Promise<string | null> {
    const context = `Austin is the Capital of Texas`;
    try {
      // Tokenize the question and context
      const tokenizedQuestion = this.tokenizer!.tokenize(question);
      const tokenizedContext = this.tokenizer!.tokenize(context);

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
      const startLogitsTensor = modelResponse["start_logits"] as TypedTensor<"float32">;
      const endLogitsTensor = modelResponse["end_logits"] as TypedTensor<"float32">;

      const startLogits = this.softmax(startLogitsTensor.data);
      const endLogits = this.softmax(endLogitsTensor.data);

      // Create undesired tokens mask (e.g., CLS, SEP, and padding tokens)
      const undesiredTokens = new Uint8Array(inputTokens.length);
      undesiredTokens.fill(0);
      undesiredTokens[0] = 1; // Mark CLS as undesired
      undesiredTokens[tokenizedQuestion.length + 1] = 1; // Mark first SEP as undesired
      undesiredTokens[tokenizedQuestion.length + tokenizedContext.length + 2] = 1; // Mark second SEP as undesired

      const bestSpan = this.decodeSpans(startLogits, endLogits, 1, 30, undesiredTokens);

      if (!bestSpan || bestSpan.starts.length === 0 || bestSpan.ends.length === 0) {
        console.error("No valid answer found");
        return null;
      }

      // Extract the tokens within the best span
      const answerTokens = inputTokens.slice(bestSpan.starts[0], bestSpan.ends[0] + 1);

      // Decode the answer tokens back to string, excluding any special tokens
      const answer = this.tokenizer!.decode(answerTokens.filter(token => token !== SEP_INDEX && token !== CLS_INDEX));

      console.log("Best span: ", bestSpan.starts[0], bestSpan.ends[0]);
      console.log("Answer tokens: ", answerTokens);
      console.log("Answer final: ", answer);
      
      return answer;
    } catch (error) {
      console.error("TransformerOps: Failed to process question", error);
      return null;
    }
  }
}
