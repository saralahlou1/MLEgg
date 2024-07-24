/*
* responsible for handling aspects of the language wrt egg from parsing in the structs to
* outputting structs again
*/

use egg::*;

// TODO if issue decribed in cost model is solved
// it would be a good idea to add a dummy root that has as children 
// all roots of all independant expression instead of optimizing each on its own
// optimizing each on its own may not give the ideal program if some share the same children

// language definition
define_language! {
    pub enum MLIR {
        // language definition here

        // the first 2 ids are the arguments, 
        // the 3rd refers to oldId (id of the mlir value for the result in the map in the mlir pass),
        // the 4th refers to oldOpId (id of the mlir value of the operation in the map in the mlir pass)
        "linalg.matmul" = MatMul([Id; 4]),

        // same as matmul 
        "linalg.dot" = Dot([Id; 4]),

        // the first is the argument
        // the second oldId
        // the third oldOpId
        "linalg.transpose" = Transpose([Id; 3]),

        // same as matmul
        "linalg.add" = Add([Id; 4]),

        // same as transpose
        "tensor.extract_slice" = ExtractSlice([Id; 3]),

        // matrix will be constructed such that its first 2 children are always the dimensions. 
        // the 3rd child is oldId
        // the 4th is oldOpId
        "matrix" = Matrix([Id; 4]),

        // this is just a scalar to represent the dimensions for matrices
        // dimensions for other operations will be infered with our e-class analysis
        Num(i32),
    }
}

// set of rules
pub fn make_rules() -> Vec<Rewrite<MLIR, Dimensions>> {
    vec![
        // rewrite! rules here
        // for equivilence, write both ways

        // associativity of matmul
        rewrite!("commute1-matmul"; "(linalg.matmul ?a (linalg.matmul ?b ?c ?d ?f) ?e ?g)" => "(linalg.matmul (linalg.matmul ?a ?b ?d ?d) ?c ?e ?e)"),
        rewrite!("commute2-matmul"; "(linalg.matmul (linalg.matmul ?a ?b ?d ?f) ?c ?e ?g)" => "(linalg.matmul ?a (linalg.matmul ?b ?c ?d ?d) ?e ?e)") ,
        
        // transformation from dot product to matmul and from matmul to dot
        // note that the later works only if it's between vectors
        // we introduce a conditional for the rewrite for this purpose
        // the -1 is a temporary value for oldId and oldOpId
        // this is because these new ops don't have any corresponding 
        // id that already exists and is not taken
        // it can cause issues if it remains in the final expression if it remains
        // this will be taken care of later when transforming the optimized expr back to a graph
        // note: the -1 usually wont remain there since we normally won't introduce a transpose 
        // for nothing since its costy. when we apply this opt, it is to remove the transpose

        // we solve this issue anyways to keep a robus code in case of any expension
        rewrite!("dot to matmul"; "(linalg.dot ?a ?b ?c ?d)" => "(linalg.matmul (linalg.transpose ?a -1 -1) ?b ?c ?d)"),
        rewrite!("matmul to dot"; "(linalg.matmul ?a ?b ?c ?d)" => "(linalg.dot (linalg.transpose ?a -1 -1) ?b ?c ?d)" 
        if is_vector_transposed("?a") if is_vector("?b")),

        // double transpose = id
        // not doing the other implication since it won't be useful
        // We express the different combinations possible
        // it will be a new one for each new operation
        // notice that old_op_id changes for the inner most section
        // it now stores the old_op_id of the outer most transpose
        // that's why we need to do this rewrite for each op
        // (to update old op id for the innermost argument by the one of the outer transpose)
        rewrite!("double transpose"; "(linalg.transpose(linalg.transpose (matrix ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(matrix ?a ?b ?c ?h)"),
        rewrite!("double transpose1"; "(linalg.transpose(linalg.transpose (linalg.matmul ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(linalg.matmul ?a ?b ?c ?h)"),
        rewrite!("double transpose2"; "(linalg.transpose(linalg.transpose (linalg.dot ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(linalg.dot ?a ?b ?c ?h)"),
        rewrite!("double transpose3"; "(linalg.transpose(linalg.transpose (linalg.add ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(linalg.add ?a ?b ?c ?h)"),


        // this is temporary. Maybe a good idea would be to add an operation to extend the dim by one
        // and that op would cancels extract_slice. we extend the dim by 1 when switching from dot to matmul
        // we reduce when going from matmul to dot

        // I am handling any issues for this case when writing back automatically
        // by checking the types and creating operations that respect the original types
        rewrite!("mini test"; "(tensor.extract_slice ?a ?b ?c)" => "?a"),

        // distributivity for addition
        // the problem for -1 is solved when reconstructing the graph
        rewrite!("distributive-matmul"; "(linalg.matmul ?a (linalg.add ?b ?c ?d ?f) ?e ?g)" => "(linalg.add (linalg.matmul ?a ?b -1 -1) (linalg.matmul ?a ?c -1 -1) ?e ?e)"),
        
        // factorisation
        rewrite!("distributive2-matmul"; "(linalg.add (linalg.matmul ?a ?b ?d ?g) (linalg.matmul ?a ?c ?f ?h) ?e ?i)" => "(linalg.matmul ?a (linalg.add ?b ?c ?d ?d) ?e ?e)") ,
        rewrite!("distributive3-matmul"; "(linalg.add (linalg.matmul ?a ?b ?d ?g) (linalg.matmul ?c ?b ?f ?h) ?e ?i)" => "(linalg.matmul (linalg.add ?a ?c ?d ?d) ?b ?e ?e)") ,
    ]
}

// this function implements the condition trait
// it is used to check if an argument is a vector for rewrites
fn is_vector(var: &str) -> impl Fn(&mut EGraph<MLIR, Dimensions>, Id, &Subst) -> bool {
    // we parse the given expr
    let var: Var = var.parse().unwrap();
    move |egraph, _root, subst: &Subst| {
        let id = subst[var];
        // we check if in the data section, the dimension correspond to what's wanted
        // we want the second dimension to be 1 for the argument to be a vector
        egraph[id].data[1] == 1
    }
}

// the logic is the same as the function above
fn is_vector_transposed(var: &str) -> impl Fn(&mut EGraph<MLIR, Dimensions>, Id, &Subst) -> bool {
    let var: Var = var.parse().unwrap();
    move |egraph, _root, subst: &Subst| {
        let id = subst[var];
        egraph[id].data[0] == 1
    }
}

// e-class analysis: will keep track of the dimensions through the operations 
#[derive(Default)]
pub struct Dimensions;

impl Analysis<MLIR> for Dimensions{
    // I chose the Data type to be an array of size 2.
    // First entery for nb of rows, second for nb columns
    type Data = [i32; 2];

    fn make(egraph: &EGraph<MLIR, Self>, enode: &MLIR) -> Self::Data {
        // getter function. Given an id, it gives back its data section
        let get = |id: &Id| &egraph[*id].data;

        match enode {
            // A x B = C, A is nxm and B is mxk => C is nxk
            MLIR::MatMul([a, b, _c, _d]) => [get(a)[0], get(b)[1]],

            // dot gives back a scalar
            MLIR::Dot([_a, _b, _c, _d]) => [1,1],

            // For transpose, we simply invert the dims of what's inside
            MLIR::Transpose([a, _b, _c]) => [get(a)[1], get(a)[0]],

            // The dims don't change after addition
            MLIR::Add([a, _b, _c, _d]) => [get(a)[0], get(a)[1]],

            // for now extract slice is only used to reduce the dimensions
            // by 1 to be able to do dot product. So we keep the dims
            // as they are in the analysis to reason more easily
            MLIR::ExtractSlice([a, _b, _c]) => [get(a)[0], get(a)[1]],

            // matrix will always have as its first two args the dims
            // which are expressed as Num(i32)
            // We just take the value of the num
            MLIR::Matrix([a,b, _c, _d]) =>  [get(a)[0], get(b)[0]],

            // numbers will always represent dimensions in our language
            // we fix a convension of storing the value of Num in the
            // first sloth of the array to be able to extract it later when needed.
            // Note: for Num, we are only interested in the actual value stored inside
            // since this is what's used to determine the dims of matrices as seen above
            MLIR::Num(a) => [*a,0],
        }
    }

    // for now we won't need this function I think
    // it would be a good idea to ask someone familiar with the fct
    fn merge(&mut self, _to: &mut Self::Data, _from: Self::Data) -> DidMerge {
        DidMerge(false, false)
    }
}


// here we implement our own cost function in order to chose the best expression
// the cost function records the number of computations needed to do the op
// we also add the number of loops needed to implement the op
// this makes the cost function accurate for our language
struct DimCostFn<'a>{
    egraph: &'a EGraph<MLIR, Dimensions>,
}

impl<'a> CostFunction<MLIR> for DimCostFn<'a> {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &MLIR, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        // here we define the cost of each operation
        let op_cost = match enode {
            // nxm times lxk has around nxkxl operations and 3 loops
            MLIR::MatMul([a, b, _c, _d]) => {
                let a_dim = self.egraph[*a].data;
                let b_dim = self.egraph[*b].data;
                a_dim[0] * a_dim[1] * b_dim[1] + 3
            }

            // nx1 dot nx1 around n operations and 1 loop
            MLIR::Dot([a, b, _c, _d]) => {
                let a_dim = self.egraph[*a].data;
                let b_dim = self.egraph[*b].data;
                a_dim[0] * b_dim[1] + 1
            }

            // we need to add every element so the cost is nxm and 2 loops
            MLIR::Add([a, _b, _c, _d]) => {
                let a_dim = self.egraph[*a].data;
                a_dim[0] * a_dim[1] + 2
            }

            // to transpose, we need to read then write every element of the matrix
            // thus, the cost is nxm plus 2 loops
            MLIR::Transpose([a, _b, _c]) => {
                let a_dim = self.egraph[*a].data;
                a_dim[0] * a_dim[1] + 2
            }

            // for now we assume that this operation is free
            MLIR::ExtractSlice([_a, _b, _c]) => 0,

            // getting the matrix shouldn't cost us anything
            // performing operations on it is what costs
            MLIR::Matrix([_a, _b, _c, _d]) => 0,

            // num is just used to store matrix dims so cost=0
            MLIR::Num(_a) => 0,
        };

        // The final cost will be the cumulative cost
        // TODO: if an expression is used multiple times, we should only charge the cost once
        // ex: AxB + AxB. the cost model should count the cost of AxB once
        return enode.fold(op_cost, |sum, id| sum + costs(id));
    }
}


// this is a funtion for which given an expr in our lang, gives back the best expr using our cost function
pub fn optimise(start: &RecExpr<MLIR>) -> (i32, RecExpr<MLIR>) {

    // Step 1: Construct an EGraph with an initial expression
    let mut egraph: EGraph<MLIR, Dimensions> = EGraph::<MLIR, Dimensions>::default();
    egraph.add_expr(&start);

    // this block is only for printing old information 
    // to compare new expr and cost with old expr and cost
    {
    // we apply the cost function to an egraph containg only the starting op
    let cost_fn = DimCostFn { egraph: &egraph };
    let extractor = Extractor::new(&egraph, cost_fn);
    // we initialise a dummy runner that won't do anything
    let runner = Runner::<MLIR, Dimensions, ()>::default().with_explanations_enabled().with_expr(&start);
    let (cost, expr) = extractor.find_best(runner.roots[0]);

    println!("original expression: {} with cost: {}", expr, cost);
    }

    // Step 2: Perform equality saturation using the provided rewrite rules
    let runner = Runner::<MLIR, Dimensions, ()>::default().with_explanations_enabled().with_expr(&start).run(&make_rules());

    egraph = runner.egraph;

    // Step 3: Create an instance of our DimCostFn and Initialize extractor with it
    let cost_fn = DimCostFn { egraph: &egraph };
    
    let extractor = Extractor::new(&egraph, cost_fn);

    // Step 4: Extract best expression with corresponding cost
    let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    println!("best expression: {} with cost: {}", best_expr, best_cost);

    (best_cost, best_expr)
}

