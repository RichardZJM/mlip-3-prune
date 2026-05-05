## Extract Problem

`extract_problem` is used to generate the $\mathbf{X^\intercal WX}$ matrix, $\mathbf{X^\intercal Wy}$ vector, and $\mathbf{y^\intercal Wy}$ scalar. These quantities are used in the `prune` function. It also print the average number of neighbors in the dataset.

The syntax is

`extract_problem pot.mtp data.cfg xtwx.bin xtwy.bin [options]`

Where

- `pot.mtp` is the file path to the MTP file
- `data.cfg` is the file path to the dataset to extract
- `xtwx.bin` is the output file path for the $\mathbf{X^\intercal WX}$ matrix. (binary file)
- `xtwy.bin` is the output file path for the $\mathbf{X^\intercal Wy}$ vector. (binary file)
- `[options]` are additional options to specify the weights. These are the same options as those accepted by `train`.

## Calculate Loss

`calculate_loss` is used to calculate the loss of an MTP for a dataset.

The syntax is

`calculate_loss pot.mtp dataset.cfg [options]`

Where

- `pot.mtp` is the file path to the MTP file
- `data.cfg` is the file path to the dataset to calculate the loss for
  `[options]` are additional options to specify the weights. These are the same options as those accepted by `train`.

## Output

Here is the example script's output.

```
Wrote: ./out/xtwx.bin
Wrote: ./out/xtwy.bin
yTWy: 22504.333319136716455
Average number of neighbors = 19.829365
Extraction Complete!
Training Loss: 0.277373
```
