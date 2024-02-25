#[derive(Clone)]
pub struct GaConfig {
    pub populaton_size: usize,
    pub generations: i32,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            populaton_size: 50,
            generations: 50,
        }
    }
}
