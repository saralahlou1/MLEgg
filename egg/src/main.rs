use std::env;
use std::path;

mod graph;
//mod mlir;
// this is the entry point
fn main() {
    // get the input file and read it
    let args: Vec<String> = env::args().collect();

    // the in path of the file to read
    let in_path = &args[1];
    // the out path of the file to write
    let out_path = &args[2];

    // parse the dot and convert it into a list of structs
    let graph = graph::Graph::from_file(path::Path::new(in_path));
    // conver the structs into egg exprs
    for node in graph.nodes {
        egg;
    }

    // equality saturate the expressions using the defined rules

    // convert the egg exprs into structs

    // conver the structs back into json, writing the file out
}
