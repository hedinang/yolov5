from torch import nn


model = nn.Sequential(
    (0): Focus(
      (conv): Conv(
        (conv): Conv2d(12, 80, kernel_size=(3, 3),
         stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
    )
    (1): Conv(
      (conv): Conv2d(80, 160, kernel_size=(3, 3),
       stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (2): C3(
      (cv1): Conv(
        (conv): Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(80, 80, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(80, 80, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(80, 80, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(80, 80, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(80, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(160, 320, kernel_size=(3, 3),
       stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (4): C3(
      (cv1): Conv(
        (conv): Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (4): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (5): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (6): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (7): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (8): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (9): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (10): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (11): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(320, 640, kernel_size=(3, 3),
       stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (6): C3(
      (cv1): Conv(
        (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (4): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (5): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (6): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (7): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (8): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (9): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (10): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (11): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(640, 1280, kernel_size=(3, 3),
       stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(1280, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (8): SPP(
      (cv1): Conv(
        (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1280, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): ModuleList(
        (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
      )
    )
    (9): C3(
      (cv1): Conv(
        (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1280, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (10): Conv(
      (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (11): Upsample(scale_factor=2.0, mode=nearest)
    (12): Concat()
    (13): C3(
      (cv1): Conv(
        (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (14): Conv(
      (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (15): Upsample(scale_factor=2.0, mode=nearest)
    (16): Concat()
    (17): C3(
      (cv1): Conv(
        (conv): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(160, 160, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(160, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (18): Conv(
      (conv): Conv2d(320, 320, kernel_size=(3, 3),
       stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (19): Concat()
    (20): C3(
      (cv1): Conv(
        (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(320, 320, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(320, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (21): Conv(
      (conv): Conv2d(640, 640, kernel_size=(3, 3),
       stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
       affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (22): Concat()
    (23): C3(
      (cv1): Conv(
        (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv3): Conv(
        (conv): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1280, eps=0.001, momentum=0.03,
         affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
        (3): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(640, 640, kernel_size=(3, 3),
             stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(640, eps=0.001, momentum=0.03,
             affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(320, 36, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(640, 36, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(1280, 36, kernel_size=(1, 1), stride=(1, 1))
      )
    )
