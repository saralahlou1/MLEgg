use std::env;
use std::path;

mod expr;
mod graph;
mod mlir;

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
    let expr = expr::from_dag(&graph);
    println!("{}", expr);

    // equality saturate the expressions using the defined rules
    let rules = mlir::make_rules(); // for ownership reasons, i think
    let runner = egg::Runner::default().with_expr(&expr).run(&rules);

    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize); // change optimization function as necessary
    let (_, best_expr) = extractor.find_best(runner.roots[0]);

    // convert the egg exprs into structs
    let result = expr::to_dag(&best_expr);

    // conver the structs back into json, writing the file out
    result.to_file(path::Path::new(out_path));

    // // create a new EGraph, so we can print it as Dot.
    // // TODO: is this any good?
    // let mut output: EGraph<mlir::MLIR, ()> = EGraph::default();
    // output.add_expr(&best_expr);
    // output
    //     .dot()
    //     .to_dot(path::Path::new(out_path))
    //     .expect("Couldn't write output file");
}
