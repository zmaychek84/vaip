<!--
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the MIT License.
 -->

# to list available test


```
ctest --test-dir $BUILD/vaip/unit-test -N
ctest --test-dir $BUILD/vaip/unit-test --show-only
```



# to run a single test

```
ctest --test-dir $BUILD/vaip/unit-test -R TesOnnxRunner.Main --verbose
ctest --test-dir $BUILD/vaip/unit-test --tests-regex TesOnnxRunner.Main --verbose
```

it is important to add `--verbose` otherwise we cannot see any console log.
