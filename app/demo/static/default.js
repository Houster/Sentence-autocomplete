$(function(){

    function addMessage(message, user_class, content_class, container_class) {
        let outer_div = document.createElement("div");
        let inner_div = document.createElement("div");

        inner_div.innerHTML = message;
        inner_div.classList.add(user_class, content_class);

        outer_div.appendChild(inner_div);
        outer_div.classList.add("msg", container_class);

        let container = document.getElementById("container");
        container.appendChild(outer_div);

        outer_div.scrollIntoView(false);
    }

    function addUserMessage(message) {
        addMessage(message, "user2", "msg_right_content", "msg_right");
    }

    function addServerMessage(message) {
        addMessage(message, "user1", "msg_left_content", "msg_left");
    }

    $("#form-reply").submit(function(e){

        let userMessage = $("#form-reply-input").val();
        
        addUserMessage(userMessage);

		$.ajax({
			url: "/reply",
			data: $("form").serialize(),
			type: "POST",
			success: function(response){
			    let data = JSON.parse(response);
				console.log(data.predictions);
				addServerMessage(data.predictions);
				$("#form-reply-input").val("");
			},
			error: function(error){
				console.log(error);
			}
		});

        e.preventDefault();
		return false;
	});
});