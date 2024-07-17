use std::env;
use std::path;
use egg::*;
use crate::mlir::MLIR;

mod expr;
mod graph;
mod mlir;

// this is the entry point from our mlir pass
fn main() {
    // get the input file and read it
    let args: Vec<String> = env::args().collect();

    // the in path of the file to read
    let in_path = &args[1];
    // the out path of the file to write
    let out_path = &args[2];

    // parse the dot and convert it into a graph
    let graph = graph::Graph::from_file(path::Path::new(in_path));

    // conver the graph into a list of egg exprs
    // the list contains all the independant exprs
    // TODO: we can try puting all these exps under a dummy root
    // explained in mlir.rs
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
    
    // convert the list of egg exprs back to a graph
    let result = expr::to_dag(&best_expr_list);

    // conver the graph back into dot format, writing the file out
    result.to_file(path::Path::new(out_path));
    
}
