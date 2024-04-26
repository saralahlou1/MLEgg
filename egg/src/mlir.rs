/*
* responsible for handling aspects of the language wrt egg from parsing in the structs to
* outputting structs again
*/
// imports (figure out which ones are needed)
use egg::*;

// language definition
define_language! {
    pub enum MLIR {
        // language definition here
        "linalg.matmul" = MatMul([Id; 2]),
        Var(i32),
        Symbol(Symbol),
        //Other(Symbol, Vec<Id>),
    }
}

// set of rules
pub fn make_rules() -> Vec<Rewrite<MLIR, ()>> {
    vec![
        // rewrite! rules here
        rewrite!("commute-matmul"; "(linalg.matmul ?a (linalg.matmul ?b ?c))" => "(linalg.matmul (linalg.matmul ?a ?b) ?c)"),
    ]
}
