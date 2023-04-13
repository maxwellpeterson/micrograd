use std::ops::{Add, Mul, Sub};
use std::{cell::RefCell, rc::Rc};

// Watch out! Value::clone() is not a deep clone.
#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    // Creates a leaf node in the computation DAG.
    pub fn leaf(data: f64) -> Self {
        ValueInternal::new(data, History::Leaf).into()
    }

    pub fn data(&self) -> f64 {
        let internal = self.0.borrow();
        internal.data
    }

    pub fn set_data(&self, data: f64) {
        let mut internal = self.0.borrow_mut();
        internal.data = data;
    }

    pub fn grad(&self) -> f64 {
        let internal = self.0.borrow();
        internal.grad
    }

    pub fn zero_grad(&self) {
        let mut internal = self.0.borrow_mut();
        // Unmark this node, so that it will be revisited the next time
        // Value::backward() is called.
        internal.mark = Mark::Unvisited;
        internal.grad = 0.0;
    }

    pub fn backward(&self) {
        // Reversed topological sort of nodes in the computation DAG.
        let mut topo: Vec<Value> = Vec::new();
        // Topological sort based on depth-first search
        // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        fn visit(acc: &mut Vec<Value>, value: Value) {
            // If there are cycles in the graph, this borrow_mut() will panic.
            // However, I don't think it's possible to accidentally create a
            // cycle with the current API.
            let mut internal = value.0.borrow_mut();
            if internal.mark == Mark::Unvisited {
                internal.mark = Mark::Visited;
                internal.children().for_each(|child| visit(acc, child));
                acc.push(value.clone());
            }
        }
        visit(&mut topo, self.clone());

        {
            // Need to drop the RefMut before the iteration below
            let mut internal = self.0.borrow_mut();
            // Initialize the gradient at the root of the DAG
            internal.grad = 1.0;
        }

        topo.into_iter().rev().for_each(|value| {
            value.0.borrow_mut().local_backward();
        });
    }

    pub fn pow(&self, exponent: f64) -> Self {
        let base = self.data();
        ValueInternal::new(
            base.powf(exponent),
            History::Unary(Child {
                value: self.clone(),
                local_grad: exponent * base.powf(exponent - 1.0),
            }),
        )
        .into()
    }

    pub fn exp(&self) -> Self {
        let result = self.data().exp();
        ValueInternal::new(
            result,
            History::Unary(Child {
                value: self.clone(),
                local_grad: result,
            }),
        )
        .into()
    }

    pub fn relu(&self) -> Self {
        ValueInternal::new(
            self.data().max(0.0),
            History::Unary(Child {
                value: self.clone(),
                local_grad: if self.data() > 0.0 { 1.0 } else { 0.0 },
            }),
        )
        .into()
    }

    pub fn tanh(&self) -> Self {
        let result = self.data().tanh();
        ValueInternal::new(
            result,
            History::Unary(Child {
                value: self.clone(),
                local_grad: 1.0 - result * result,
            }),
        )
        .into()
    }

    pub fn sigmoid(&self) -> Self {
        let result = 1.0 / (1.0 + (-self.data()).exp());
        ValueInternal::new(
            result,
            History::Unary(Child {
                value: self.clone(),
                local_grad: result * (1.0 - result),
            }),
        )
        .into()
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ValueInternal::new(
            self.data() + rhs.data(),
            History::Binary(
                Child {
                    value: self,
                    local_grad: 1.0,
                },
                Child {
                    value: rhs,
                    local_grad: 1.0,
                },
            ),
        )
        .into()
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs * Value::leaf(-1.0)
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let left = self.data();
        ValueInternal::new(
            left * rhs.data(),
            History::Binary(
                Child {
                    value: self,
                    local_grad: rhs.data(),
                },
                Child {
                    value: rhs,
                    local_grad: left,
                },
            ),
        )
        .into()
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl From<ValueInternal> for Value {
    fn from(internal: ValueInternal) -> Self {
        Self(Rc::new(RefCell::new(internal)))
    }
}

struct ValueInternal {
    data: f64,
    grad: f64,
    mark: Mark,
    history: History,
}

impl ValueInternal {
    fn new(data: f64, history: History) -> Self {
        Self {
            data,
            grad: 0.0,
            mark: Mark::Unvisited,
            history,
        }
    }

    fn children(&self) -> impl Iterator<Item = Value> + '_ {
        ChildIterator {
            history: &self.history,
            index: 0,
        }
    }

    // Applies one step of the chain rule between this node and its children.
    fn local_backward(&mut self) {
        match &self.history {
            History::Binary(left, right) => {
                left.value.0.borrow_mut().grad += left.local_grad * self.grad;
                right.value.0.borrow_mut().grad += right.local_grad * self.grad;
            }
            History::Unary(child) => {
                child.value.0.borrow_mut().grad += child.local_grad * self.grad;
            }
            History::Leaf => {}
        }
    }
}

struct ChildIterator<'a> {
    history: &'a History,
    index: usize,
}

impl<'a> Iterator for ChildIterator<'a> {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match (self.index, &self.history) {
            (0, History::Unary(child)) => Some(child),
            (0, History::Binary(left, _)) => Some(left),
            (1, History::Binary(_, right)) => Some(right),
            _ => None,
        };
        self.index += 1;
        result.map(|child| child.value.clone())
    }
}

#[derive(PartialEq)]
enum Mark {
    Visited,
    Unvisited,
}

enum History {
    Binary(Child, Child),
    Unary(Child),
    Leaf,
}

struct Child {
    value: Value,
    // The local gradient of the parent with respect to this child.
    //
    // This value is always known at the time of parent creation, because the
    // graph does not update when leaf values change (e.g. when model parameters
    // are updated by an optimizer). Each time leaf values change, an entirely
    // new graph is created (e.g. during the next forward pass of the model).
    // This new graph recomputes the gradients of each intermediate node based
    // on the updated leaf values. This matches the implementation of
    // karpathy/micrograd, but I'm not sure this is actually what PyTorch does.
    // If the structure of the computation graph won't change between forward
    // passes (only the values stored inside the nodes of the graph change), it
    // seems wasteful to recreate the graph from scratch every time?
    local_grad: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_vals() {
        let x = &Value::leaf(2.0);
        let y = &Value::leaf(3.0);
        let sum = x + y;

        // Check forward pass
        assert_eq!(sum.data(), 5.0);

        sum.backward();

        // Check backward pass
        assert_eq!(sum.grad(), 1.0);
        assert_eq!(x.grad(), 1.0);
        assert_eq!(y.grad(), 1.0);
    }

    #[test]
    fn mul_vals() {
        let x = &Value::leaf(2.0);
        let y = &Value::leaf(3.0);
        let product = x * y;

        // Check forward pass
        assert_eq!(product.data(), 6.0);

        product.backward();

        // Check backward pass
        assert_eq!(product.grad(), 1.0);
        assert_eq!(x.grad(), 3.0);
        assert_eq!(y.grad(), 2.0);
    }

    #[test]
    fn with_relu() {
        let x = &Value::leaf(2.0);
        let y = &Value::leaf(3.0);
        let product = x * y;
        let activation = product.relu();

        // Check forward pass
        assert_eq!(activation.data(), 6.0);

        activation.backward();

        // Check backward pass
        assert_eq!(activation.grad(), 1.0);
        assert_eq!(product.grad(), 1.0);
        assert_eq!(x.grad(), 3.0);
        assert_eq!(y.grad(), 2.0);
    }

    #[test]
    fn reused_leaf() {
        let x = &Value::leaf(2.0);
        let y = &Value::leaf(3.0);
        let product = x * y;
        let product_squared = product.pow(2.0);

        // Check forward pass
        assert_eq!(product_squared.data(), 36.0);

        product_squared.backward();

        // Check backward pass
        assert_eq!(product_squared.grad(), 1.0);
        assert_eq!(product.grad(), 12.0);
        assert_eq!(x.grad(), 36.0);
        assert_eq!(y.grad(), 24.0);
    }
}
