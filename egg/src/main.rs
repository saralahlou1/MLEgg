use std::env;
use std::path;
use egg::*;
use crate::mlir::MLIR;

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
    let expr_list= expr::from_dag(&graph);

    let mut best_expr_list: Vec<RecExpr<MLIR>> = Vec::new();

    // for each expression, extract best expression using our defined cost function
    for exp in expr_list {
        let (_, best) = mlir::optimise(&exp);
        best_expr_list.push(best);
    }

    println!("the best expressions are: ");
    for best in &best_expr_list {
        println!("{}", best);
    }
    
    // convert the egg exprs into structs
    let result = expr::to_dag(&best_expr_list);

    // conver the structs back into json, writing the file out
    result.to_file(path::Path::new(out_path));
    
}
