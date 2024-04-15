/*
* responsible for handling aspects of the language wrt egg from parsing in the structs to
* outputting structs again
*/
// imports (figure out which ones are needed)

// language definition
define_language! {
    enum MLIR {
        // language definition here
        Symbol(Symbol),
        Other(Symbol, Vec<Id>),
    }
}

// set of rules
fn make_rules() -> Vec<Rewrite<SimpleLanguage, ()>> {
    vec![
        // rewrite! rules here
    ]
}
