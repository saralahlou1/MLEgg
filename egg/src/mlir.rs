/*
* responsible for handling aspects of the language wrt egg from parsing in the structs to
* outputting structs again
*/
// imports 
use egg::*;

// language definition
define_language! {
    pub enum MLIR {
        // language definition here
        "linalg.matmul" = MatMul([Id; 4]),

        "linalg.dot" = Dot([Id; 4]),

        "linalg.transpose" = Transpose([Id; 3]),

        "linalg.add" = Add([Id; 4]),

        "tensor.extract_slice" = ExtractSlice([Id; 3]),

        // matrix will be constructed such that its children are always the dimensions. 
        // The last child is the old id of the matrix
        "matrix" = Matrix([Id; 4]),

        // will store the value of the dimensions of a matrix
        Num(i32),
    }
}

// set of rules
pub fn make_rules() -> Vec<Rewrite<MLIR, Dimensions>> {
    vec![
        // rewrite! rules here
        // for equivilence, write both ways
        rewrite!("commute1-matmul"; "(linalg.matmul ?a (linalg.matmul ?b ?c ?d ?f) ?e ?g)" => "(linalg.matmul (linalg.matmul ?a ?b ?d ?d) ?c ?e ?e)"),
        rewrite!("commute2-matmul"; "(linalg.matmul (linalg.matmul ?a ?b ?d ?f) ?c ?e ?g)" => "(linalg.matmul ?a (linalg.matmul ?b ?c ?d ?d) ?e ?e)") ,
        
        // note that the other implication works only if it's between vectors
        // the -1 can cause issues if it remains in the final expression
        // it is being kept here since we normally won't introduce a transpose 
        // for nothing since its costy. when we apply this opt, it is to remove the transpose
        // should think of solving this issue
        // maybe keep a counter of highest id in dot file, here put counter + 1
        // think more
        rewrite!("dot to matmul"; "(linalg.dot ?a ?b ?c ?d)" => "(linalg.matmul (linalg.transpose ?a -1 -1) ?b ?c ?d)"),
        rewrite!("matmul to dot"; "(linalg.matmul ?a ?b ?c ?d)" => "(linalg.dot (linalg.transpose ?a -1 -1) ?b ?c ?d)" 
        if is_vector_transposed("?a") if is_vector("?b")),

        // not doing the other implication since it won't be useful
        // We express the different combinations possible
        // it will be a new one for each new operation
        // notice that old_op_id changes for the inner most section
        // it now stores the old_op_id of the outer most transpose
        rewrite!("double transpose"; "(linalg.transpose(linalg.transpose (matrix ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(matrix ?a ?b ?c ?h)"),
        rewrite!("double transpose1"; "(linalg.transpose(linalg.transpose (linalg.matmul ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(linalg.matmul ?a ?b ?c ?h)"),
        rewrite!("double transpose2"; "(linalg.transpose(linalg.transpose (linalg.dot ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(linalg.dot ?a ?b ?c ?h)"),
        rewrite!("double transpose3"; "(linalg.transpose(linalg.transpose (linalg.add ?a ?b ?c ?f) ?d ?g) ?e ?h)" => "(linalg.add ?a ?b ?c ?h)"),


        // TODO
        // this is temporary. Maybe a good idea would be to add an operation to extend the dim by one
        // and this op cancels extract_slice. we extend the dim by 1 when switching from dot to matmul
        // we reduce when going from matmul to dot
        // I am handling any issues for this case when writing back automatically so there is no harm done
        rewrite!("mini test"; "(tensor.extract_slice ?a ?b ?c)" => "?a"),

        // for addition
        // here too think of how to solve -1
        rewrite!("distributive-matmul"; "(linalg.matmul ?a (linalg.add ?b ?c ?d ?f) ?e ?g)" => "(linalg.add (linalg.matmul ?a ?b -1 -1) (linalg.matmul ?a ?c -1 -1) ?e ?e)"),
        
        rewrite!("distributive2-matmul"; "(linalg.add (linalg.matmul ?a ?b ?d ?g) (linalg.matmul ?a ?c ?f ?h) ?e ?i)" => "(linalg.matmul ?a (linalg.add ?b ?c ?d ?d) ?e ?e)") ,
        rewrite!("distributive3-matmul"; "(linalg.add (linalg.matmul ?a ?b ?d ?g) (linalg.matmul ?c ?b ?f ?h) ?e ?i)" => "(linalg.matmul (linalg.add ?a ?c ?d ?d) ?b ?e ?e)") ,
    ]
}

// The function signature output here implements the `Condition` trait in egg.
fn is_vector(var: &str) -> impl Fn(&mut EGraph<MLIR, Dimensions>, Id, &Subst) -> bool {
    // we pass in an expression and parse it
    let var: Var = var.parse().unwrap();
    move |egraph, _root, subst: &Subst| {
        let id = subst[var];
        // we check if in the data section, the dimension correspond to what's wanted
        egraph[id].data[1] == 1
    }
}

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
    type Data = [i32; 3];

    fn make(egraph: &EGraph<MLIR, Self>, enode: &MLIR) -> Self::Data {
        // getter function
        let get = |id: &Id| &egraph[*id].data;
        match enode {
            // A x B = C, A is nxm and B is mxk => C is nxk
            MLIR::MatMul([a, b, c, _d]) => [get(a)[0], get(b)[1], get(c)[0]],

            // dot gives back a scalar
            MLIR::Dot([_a, _b, c, _d]) => [1,1, get(c)[0]],

            // For transpose, we simply invert the dims of what's inside
            MLIR::Transpose([a, _b, c]) => [get(a)[1], get(a)[0], get(c)[0]],

            // The dims don't change after addition
            MLIR::Add([a, _b, c, _d]) => [get(a)[0], get(a)[1], get(c)[0]],

            // for now extract slice is only used to reduce the dimension 
            // by 1 to be able to do the dot product. So we keep the dims 
            // as they are in the analysis to reason more easily
            MLIR::ExtractSlice([a, b, _c]) => [get(a)[0], get(a)[1], get(b)[0]],

            // matrix will always contain ids that correspond to Num.
            // We just take the value of the num
            MLIR::Matrix([a,b, c, _d]) =>  [get(a)[0], get(b)[0], get(c)[0]],

            // numbers will always represent dimensions in our language
            // we fix a convension of storing the value of Num in the
            // first sloth of the array. 
            // Note: for Num, we are only interested in the actual value stored
            MLIR::Num(a) => [*a,0,0],
        }
    }

    // for now we won't need this function I think
    fn merge(&mut self, _to: &mut Self::Data, _from: Self::Data) -> DidMerge {
        DidMerge(false, false)
    }
}


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
        // we add the number of loops to the cost as well
        let op_cost = match enode {
            // nxm times lxk has around nxkxl operations
            MLIR::MatMul([a, b, _c, _d]) => {
                let a_dim = self.egraph[*a].data;
                let b_dim = self.egraph[*b].data;
                a_dim[0] * a_dim[1] * b_dim[1] + 3
            }

            // nx1 dot nx1 around n^2 operations
            MLIR::Dot([a, b, _c, _d]) => {
                let a_dim = self.egraph[*a].data;
                let b_dim = self.egraph[*b].data;
                a_dim[0] * b_dim[1] + 1
            }

            // we need to add every element so the cost is nxm
            MLIR::Add([a, _b, _c, _d]) => {
                let a_dim = self.egraph[*a].data;
                a_dim[0] * a_dim[1] + 2
            }

            // to transpose, we need to read then write every element of the matrix
            // thus, the cost is nxm
            MLIR::Transpose([a, _b, _c]) => {
                let a_dim = self.egraph[*a].data;
                a_dim[0] * a_dim[1] + 2
            }

            // for now we assume that this operation is free
            MLIR::ExtractSlice([_a, _b, _c]) => 0,

            // getting the matrix shouldn't cost us anything
            MLIR::Matrix([_a, _b, _c, _d]) => 0,

            // num is just used to store matrix dims so cost=0
            MLIR::Num(_a) => 0,
        };

        // The final cost will be the cumulative cost
        return enode.fold(op_cost, |sum, id| sum + costs(id));
    }
}


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

    // Step 3: Create a DimCostFn instance  and Initialize extractor with our DimCostFn
    let cost_fn = DimCostFn { egraph: &egraph };
    
    let extractor = Extractor::new(&egraph, cost_fn);

    // Extract best expression with corresponding cost
    let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    println!("best expression: {} with cost: {}", best_expr, best_cost);

    (best_cost, best_expr)
}


egg::test_fn! {
    id_transpose_nested, 
    make_rules(),            
    "(linalg.matmul (linalg.transpose (linalg.transpose (matrix 1 2 0 0) 1 1) 2 2) (matrix 2 1 3 3) 4 4)"   
    =>
    "(linalg.matmul (matrix 1 2 0 2) (matrix 2 1 3 3) 4 4)"  
    
 }

 egg::test_fn! {
    id_transpose_nested2, 
    make_rules(),            
    "(linalg.add (linalg.transpose (linalg.transpose (linalg.add (matrix 2 3 1 1) (matrix 2 3 1 1) 0 0) 3 3) 5 5) (linalg.transpose (linalg.transpose (linalg.add (matrix 2 3 1 1) (matrix 2 3 1 1) 0 0) 3 3) 5 5) 7 7)"   
    =>
    "(linalg.add (linalg.add (matrix 2 3 1 1) (matrix 2 3 1 1) 0 5) (linalg.add (matrix 2 3 1 1) (matrix 2 3 1 1) 0 5) 7 7)"  
    
 }

 egg::test_fn! {
    id_transpose_nested3, 
    make_rules(),            
    "(linalg.transpose (linalg.transpose (linalg.add (matrix 2 3 1 1) (matrix 2 3 1 1) 0 0) 3 3) 5 5)"   
    =>
    "(linalg.add (matrix 2 3 1 1) (matrix 2 3 1 1) 0 5)"  
    
 }