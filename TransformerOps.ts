import { Asset } from "expo-asset";
import { InferenceSession, TypedTensor } from "onnxruntime-react-native";
import { createModelInput } from "./tokenizer/BertProcessor";
import { BertTokenizer, CLS_INDEX, SEP_INDEX, loadTokenizer } from "./tokenizer/BertTokenizer";
import { SEP_TOKEN } from "./assets/tokenizer/BertTokenizer";
type TokenizeOptions = {
  padding?: boolean;
  truncation?: boolean;
  maxLength?: number;
};
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

  max(arr) {
    if (arr.length === 0) throw Error('Array must not be empty');
    let max = arr[0];
    let indexOfMax = 0;
    for (let i = 1; i < arr.length; ++i) {
        if (arr[i] > max) {
            max = arr[i];
            indexOfMax = i;
        }
    }
    return [Number(max), indexOfMax];
}

  softmax(arr) {
    // Compute the maximum value in the array
    const maxVal = this.max(arr)[0];

    // Compute the exponentials of the array values
    const exps = arr.map(x => Math.exp(x - maxVal));

    // Compute the sum of the exponentials
    // @ts-ignore
    const sumExps = exps.reduce((acc, val) => acc + val, 0);

    // Compute the softmax values
    const softmaxArr = exps.map(x => x / sumExps);

    return /** @type {T} */(softmaxArr);
}

product(array1: Array<any>, array2: Array<any>): Array<[any, any]> {
  const result: Array<[any, any]> = [];
  for (let item1 of array1) {
      for (let item2 of array2) {
          result.push([item1, item2]);
      }
  }
  return result;
}

tokenizeQuestionAndContext(
  tokenize: (text: string, options?: TokenizeOptions) => string[],
  question: string,
  context: string,
  options: TokenizeOptions = {}
) {
  // Default options for tokenization
  const { padding = true, truncation = true, maxLength = 512 } = options;

  // Tokenize question and context as a text pair
  let questionTokens = this.tokenizer!.tokenize(question);
  let contextTokens = this.tokenizer!.tokenize(context);

  // Add special tokens if required (this example assumes a BERT-like model)
  const clsToken = '[CLS]';
  const sepToken = '[SEP]';

  // Construct token sequence: [CLS] question tokens [SEP] context tokens [SEP]
  const inputTokens = [clsToken, ...questionTokens, sepToken, ...contextTokens, sepToken];

  // If truncation is enabled, truncate question and context tokens
  if (truncation) {
    const totalLength = questionTokens.length + contextTokens.length + 3; // Adding 3 for [CLS] and [SEP] tokens
    if (totalLength > maxLength) {
        // Allocate space for question and context based on proportion
        const questionLength = Math.floor((maxLength - 3) * 0.5); // Allow half the space for question
        const contextLength = maxLength - 3 - questionLength; // Remaining for context
        questionTokens = questionTokens.slice(0, questionLength);
        contextTokens = contextTokens.slice(0, contextLength);
    }
}

  // Apply padding if necessary
  if (padding) {
      const maxLength = options.maxLength || 512; // Adjust this based on model's max input length
      while (inputTokens.length < maxLength) {
          inputTokens.push('[PAD]');
      }
  }

  return inputTokens;
}
public async runInference(question: string, topk = 1): Promise<any> {
  const context = `
  
Save 3% on all your Apple purchases with Apple Card.1 Apply and use in minutes2
APPLE ACCOUNT
amanueltaddesse19@gmail.com	BILLED TO
Visa .... 5006
Amanuel Taddesse
5100 Gadsden Ave
Keller , TX 76244
USA
DATE
Aug 31, 2024
ORDER ID
MSHLQYB2M8	DOCUMENT NO.
206845010900
App Store
	Peacock TV: Stream TV & Movies
Peacock Premium (Monthly)
Renews Sep 27, 2024
Report a Problem	
$7.99
Subtotal	 	$7.99
 
Tax	 	$0.66
TOTAL		$8.65
 
Apple Card
Save 3% on all your Apple purchases.
Apply and use in minutes
 

  `;  // Example context
  try {
      const tokenizedQuestion = this.tokenizer!.tokenize(question);
      const tokenizedContext = this.tokenizer!.tokenize(context);
      const inputs = [
        CLS_INDEX,        
          ...tokenizedQuestion, 
          SEP_INDEX, 
          ...tokenizedContext, 
          SEP_INDEX
      ];

      // Create ONNX input tensors
      const input = await createModelInput(inputs);

      // Run the model and get start and end logits
      const output: InferenceSession.ReturnType = await this.onnxModel!.run(input);
      console.log("Model response: ", output);

      console.log("Model response: ", output);

    /** @type {QuestionAnsweringOutput[]} */
    const toReturn = [];
    for (let j = 0; j < output.start_logits.dims[0]; ++j) {
        const sepIndex:number = SEP_INDEX;

        // Extract start and end logits from cpuData
        const s1 = Array.from(this.softmax(output.start_logits.cpuData))
            .map((x, i) => [x, i])
            .filter(x => (x[1] as number) > 6);
        const e1 = Array.from(this.softmax(output.end_logits.cpuData))
            .map((x, i) => [x, i])
            .filter(x => (x[1] as number) > 6);

        console.log("S1 and E1", s1, e1);

        const options = this.product(s1, e1)
            .filter(x => x[0][1] <= x[1][1])
            .map(x => [x[0][1], x[1][1], x[0][0] * x[1][0]])
            .sort((a, b) => b[2] - a[2]);

        for (let k = 0; k < Math.min(options.length, topk); ++k) {
            let [start, end, score] = options[k];
            const start_ = start - 1;
            const end_ = end - 1;
            const answer_tokens = [...inputs].slice(start_, end_ + 1);

            const answer = this.tokenizer.decode(answer_tokens);

           // TODO add start and end?
                // NOTE: HF returns character index
                toReturn.push({
                  answer, score
              });
        }
    }
    return (topk === 1) ? toReturn[0].answer : toReturn;
  } catch (error) {
      console.error("TransformerOps: Failed to process question", error);
      return null;
  }
  }
}
