pub trait ContextProvider {
    fn title(&self) -> &str;
    fn get_info(&self) -> String;
}
