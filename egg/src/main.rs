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
    let expr = expr::Expr::from_dag(&graph);

    // equality saturate the expressions using the defined rules
    let rules = mlir::make_rules(); // for ownership reasons, i think
    let runner = egg::Runner::default().with_expr(&expr).run(&rules);

    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize); // change optimization function as necessary
    let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

    // convert the egg exprs into structs
    let result = expr::Expr::to_dag(&best_expr);

    // conver the structs back into json, writing the file out
    graph::Graph::to_file(path::Path::new(out_path), &result);
}
