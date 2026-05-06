FROM rust:1.93.1-alpine

COPY . .

RUN cargo build
ENTRYPOINT [ "cargo", "run" ]
