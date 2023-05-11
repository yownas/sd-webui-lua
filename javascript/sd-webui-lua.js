function submit_sd_webui_lua(){
    var id = randomId()
    requestProgress(id, gradioApp().getElementById('sd_webui_lua_results'), null, function(){})

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}
