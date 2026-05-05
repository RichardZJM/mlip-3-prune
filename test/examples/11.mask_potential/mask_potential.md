## Mask Blank

`mask_blank` is used to write out a new, unfitted MTP file given a mask. It is expected that this potential is fitted before usage, likely with multiple seeds.

The syntax is

`extract_problem base.mtp mask.csv row out.mtp`

Where

- `pot.mtp` is the file path to the base MTP file
- `mask.csv` is the file path to the masks.
- `row` is a zero-indexed integer value specifying which row of the mask file to use.
- `out.mtp` is the output file path for the new mask MTP.

Mask files are in the format output by the prune command. Each row contains the mask describing a pruned MTP. This is a sequence of `1` and `0`.

## Mask Blank

`mask_inherited` is used to write out a new, fitted MTP file given a mask. The MTP is fitted by preserving the radial parameters of the base potential and solving for the linear parameters. It is expected that this potential is refitted before usage. No seeds are need as this is a deterministic process.

The syntax is

`mask_inherited base.mtp config.json mask.csv row out.mtp`

Where

- `pot.mtp` is the file path to the base MTP file
- `config.json` is the file path to the configuration file. This is the same format as `prune` and the same file can be used.
- `mask.csv` is the file path to the masks.
- `row` is a zero-indexed integer value specifying which row of the mask file to use.
- `out.mtp` is the output file path for the new mask MTP.

Mask files are in the format output by the prune command. Each row contains the mask describing a pruned MTP. This is a sequence of `1` and `0`.

## Output

Here is the example script's output.

```
Successfully wrote blank pruned potential to out/blank.mtp
Successfully wrote inherited pruned potential to out/inherited.mtp
```
