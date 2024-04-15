use egg::*;

define_language! {
    enum MyLanguage {
        "aaa" = A([Id; 3]),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),

        Symbol(Symbol),
        Other(Symbol, Vec<Id>), // fallback for the fallback
    }
}

fn main() {
    //let rules: &[Rewrite<SymbolLang, ()>] = &[
    //    rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"),
    //    rewrite!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"),
    //
    //    rewrite!("associate-add"; "(+ (+ ?x ?y) ?z)" => "(+ ?x (+ ?y ?z))"),
    //    rewrite!("associate-mul"; "(* (* ?x ?y) ?z)" => "(* ?x (+ ?y ?z))"),
    //
    //    rewrite!("add-0"; "(+ ?x 0)" => "?x"),
    //    rewrite!("mul-0"; "(* ?x 0)" => "0"),
    //    rewrite!("mul-1"; "(* ?x 1)" => "?x"),
    //];
    //
    //let start = "(+ (* 0 (+ a b)) (* a (+ a b)))".parse().unwrap();
    //
    //// actually does ES
    //let runner = Runner::default().with_expr(&start).run(rules);
    //
    //let extractor = Extractor::new(&runner.egraph, AstSize);
    //
    //let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    //
    //println!("original expr: {}", start);
    //println!("best expr: {}", best_expr);
    
    let mut expr = RecExpr::default();
    let a = expr.add(MyLanguage::Symbol(Symbol::from("a")));
    let b = expr.add(MyLanguage::Symbol(Symbol::from("b")));
    let plus = expr.add(MyLanguage::Add([a, b]));
    let aaa = expr.add(MyLanguage::A([a, a, plus]));
    println!("expr: {}", expr);
    //let rules: &[Rewrite<MyLanguage, ()>] = &[
    //    rewrite!("commute-add");
    //];
}
