# flacpage

Times a FLAC page decode on the host with symphonia: the whole decode,
32768-frame pages with a kept reader and with a reopened file, and a check that
unaligned windows equal the slice of the whole decode.

```sh
cargo run --release --locked -- <file.flac>
```

The input is a FLAC the host encoder wrote, for example `data/mix240.flac` of
[aac_pages](../aac_pages/readme.md). It prints four lines.
