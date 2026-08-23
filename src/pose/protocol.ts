export type WorkerInboundMessage =
  | {
      type: 'init';
      modelUrl: string;
      fallbackUrl?: string;
      wasmPaths: string;
      inputWidth: number;
      inputHeight: number;
      simccSplitRatio: number;
    }
  | {
      type: 'frame';
      frameId: number;
      mediaTime: number;
      bitmap: ImageBitmap;
      trackClubHead?: boolean;
      leadIsLeft?: boolean;
    }
  | {
      type: 'dispose';
    }
  | {
      type: 'resetClub';
    };

export type WorkerOutboundMessage =
  | {
      type: 'ready';
      executionProvider: string;
      inputName: string;
      inputWidth: number;
      inputHeight: number;
    }
  | {
      type: 'pose';
      frameId: number;
      mediaTime: number;
      inferenceMs: number;
      width: number;
      height: number;
      xs: Float32Array;
      ys: Float32Array;
      scores: Float32Array;
      clubHead?: Float32Array;
    }
  | {
      type: 'error';
      message: string;
    };
