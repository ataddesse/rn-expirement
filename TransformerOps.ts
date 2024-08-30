import { InferenceSession, TypedTensor } from "onnxruntime-react-native";
import { BertTokenizer, loadTokenizer } from "./tokenizer/BertTokenizer";
import { createModelInput } from "./tokenizer/BertProcessor";
import { Asset } from "expo-asset";

export class TransformerOps {
  private onnixModel?: InferenceSession;
  private tokenizer?: BertTokenizer;
  private static instance?: TransformerOps;

  private constructor(model: InferenceSession, tokenizer: BertTokenizer) {
    this.onnixModel = model;
    this.tokenizer = tokenizer;
  }

  public static async getInstance(): Promise<TransformerOps> {
    if (!TransformerOps.instance) {
      const onnixModel = await TransformerOps.loadOnnixModel();
      const tokenizer = await loadTokenizer();
      TransformerOps.instance = new TransformerOps(onnixModel, tokenizer);
    }
    return TransformerOps.instance;
  }

  public async runInference(input: string): Promise<string | undefined> {
    if (!input) return;

    try {
      // Tokenize the input text
      const tokenizedInput = this.tokenizer!.tokenize(input);

      // Prepare the input for the ONNX model
      const onnxInput = await createModelInput(tokenizedInput);

      // Run the model inference
      const modelResponse: InferenceSession.ReturnType = await this.onnixModel!.run(onnxInput);

      // Get the logits from the model output
      const logits = modelResponse["logits"] as TypedTensor<"float32">;
      const logitsArray = logits.data as Float32Array;

      // Apply softmax to get probabilities
      const probabilities = this.softmax(logitsArray);

      // Get the index of the highest probability (argmax)
      const predictedLabelIndex = probabilities.indexOf(Math.max(...probabilities));

      // Map the predicted label index to the corresponding class label
      const idToLabel = { 0: "travel", 1: "career", 2: "financial", 3: "purchase", 4: "sales" };
      const predictedClass = idToLabel[predictedLabelIndex];

      return predictedClass;
    } catch (error) {
      console.error("TransformerOps: Inference failed for input " + input, error);
      return;
    }
  }

  private softmax(logits: Float32Array): any {
    const maxLogit = Math.max(...logits);
    const exps = logits.map((logit) => Math.exp(logit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map((exp) => exp / sumExps);
  }

  private static async loadOnnixModel(): Promise<InferenceSession> {
    const assets = await Asset.loadAsync(require("./assets/distil/model.onnx"));
    const modelUri = assets[0].localUri;

    if (!modelUri) {
      throw new Error("TransformerOps: Initialization failed, could not find distilbert-base-uncased.onnx");
    }

    const model = await InferenceSession.create(modelUri);
    return model;
  }
}
